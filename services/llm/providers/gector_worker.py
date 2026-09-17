"""Child-only GECToR runtime; framing, GPU identity, and lifecycle are shared."""
from __future__ import annotations
import json, math, tempfile
from pathlib import Path
from . import python_worker

class Runtime(python_worker.Runtime):
    def load(self):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from gector import GECToR, GECToRConfig, load_verb_dict
        import gector.modeling as modeling
        self.tokenizer = AutoTokenizer.from_pretrained(self.root, local_files_only=True, trust_remote_code=False)
        config = GECToRConfig.from_pretrained(self.root, local_files_only=True)
        # This is the configured execution bucket, independent of the encoder's
        # 512-position architecture limit and any artifact/default generation cap.
        config.max_length = self.config["max_subword_tokens"]
        base_config = {"model_type":"deberta","attention_probs_dropout_prob":0.1,"hidden_act":"gelu","hidden_dropout_prob":0.1,"hidden_size":1024,"initializer_range":0.02,"intermediate_size":4096,"max_position_embeddings":512,"relative_attention":True,"pos_att_type":"c2p|p2c","layer_norm_eps":1e-7,"max_relative_positions":-1,"position_biased_input":False,"num_attention_heads":16,"num_hidden_layers":24,"type_vocab_size":0,"vocab_size":50265}
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); (base/"config.json").write_text(json.dumps(base_config),encoding="utf-8"); config.model_id=str(base)
            old_model,old_tokenizer=modeling.AutoModel.from_pretrained,modeling.AutoTokenizer.from_pretrained
            modeling.AutoModel.from_pretrained=lambda _path,**_: AutoModel.from_config(AutoConfig.from_pretrained(base,local_files_only=True))
            modeling.AutoTokenizer.from_pretrained=lambda _path,**_: self.tokenizer
            try:
                self.model,loading=GECToR.from_pretrained(self.root,config=config,local_files_only=True,use_safetensors=True,output_loading_info=True)
                required = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
                if set(loading) != required or any(loading[key] for key in required): raise RuntimeError("GECToR weight load is incomplete")
            finally: modeling.AutoModel.from_pretrained,modeling.AutoTokenizer.from_pretrained=old_model,old_tokenizer
        if getattr(self.model.config, "max_length", None) != self.config["max_subword_tokens"]:
            raise RuntimeError("GECToR model max_length does not match bucket")
        # GECToR 1.2.0 installs its encoder at ``bert``.  Do not infer this
        # from a generic nested model attribute: the GECToR config itself has
        # no encoder position limit and would turn a valid local load into a
        # generic worker failure.
        encoder_config = getattr(getattr(self.model, "bert", None), "config", None)
        if getattr(encoder_config, "max_position_embeddings", None) != 512:
            raise RuntimeError("GECToR encoder architecture is incompatible")
        self.model=self.model.to(torch.device("cuda:0"), dtype=torch.float32); self.model.eval(); self.encode,self.decode=load_verb_dict(str(self.root/"verb-form-vocab.txt"))
    def _preprocess(self, text):
        words = ["$START"] + text.split(" ")
        add_special_tokens = not bool(getattr(self.model.config, "is_official_model", False))
        encoded = self.tokenizer([words], return_tensors="pt", max_length=self.config["max_subword_tokens"],
            padding="max_length", truncation=False, is_split_into_words=True,
            add_special_tokens=add_special_tokens)
        ids = encoded.get("input_ids")
        if ids is None or not hasattr(ids, "shape") or len(tuple(ids.shape)) != 2 or tuple(ids.shape)[0] != 1:
            raise RuntimeError("GECToR tokenizer returned malformed ids")
        if tuple(ids.shape)[1] > self.config["max_subword_tokens"]:
            return None
        if tuple(ids.shape)[1] != self.config["max_subword_tokens"]:
            raise RuntimeError("GECToR tokenizer did not pad bounded ids")
        word_ids = getattr(encoded, "word_ids", None)
        alignment = word_ids(batch_index=0) if callable(word_ids) else None
        if not isinstance(alignment, list) or len(alignment) != self.config["max_subword_tokens"] or any(
                value is not None and (type(value) is not int or value < 0 or value >= len(words)) for value in alignment):
            raise RuntimeError("GECToR tokenizer returned malformed word alignment")
        return encoded

    def _check(self,texts,keep_confidence,min_error_prob,n_iteration,batch_size):
        if (not isinstance(texts,list) or len(texts)!=1 or not isinstance(texts[0],str) or not texts[0].strip()
                or not isinstance(keep_confidence,(int,float)) or isinstance(keep_confidence,bool) or not math.isfinite(keep_confidence)
                or not isinstance(min_error_prob,(int,float)) or isinstance(min_error_prob,bool) or not math.isfinite(min_error_prob)
                or keep_confidence!=self.config["keep_confidence"] or min_error_prob!=self.config["min_error_prob"]
                or type(n_iteration) is not int or n_iteration!=self.config["max_iterations"]
                or type(batch_size) is not int or batch_size!=1): raise ValueError("request does not match GECToR bucket")
        return self._preprocess(texts[0]) is not None
    def validate(self,**values): return {"accepted": self._check(**values)}
    def execute(self,**values):
        import torch
        from gector import predict
        if not self._check(**values): raise RuntimeError("execution input exceeds no-truncation bucket")
        with torch.inference_mode(): result=predict(self.model,self.tokenizer,values["texts"],self.encode,self.decode,keep_confidence=values["keep_confidence"],min_error_prob=values["min_error_prob"],n_iteration=values["n_iteration"],batch_size=values["batch_size"])
        if not isinstance(result,list) or len(result)!=1 or not isinstance(result[0],str) or not result[0].strip(): raise RuntimeError("invalid aligned output")
        try:
            if self._preprocess(result[0]) is None:
                raise ValueError("GECToR output exceeds no-truncation bucket")
        except (ValueError, RuntimeError) as exc: raise RuntimeError("output exceeds or violates GECToR bucket") from exc
        return result

if __name__=="__main__": raise SystemExit(python_worker.main(Runtime,True))
