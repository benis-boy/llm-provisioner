"""Unit regressions for the standalone image broker boundary harness."""
from __future__ import annotations

from types import SimpleNamespace
import stat
import unittest

from tests.integration.test_image_broker_boundary import _assert_immutable, _private_listen


class ImageBrokerBoundaryHarnessTests(unittest.TestCase):
    def test_private_listener_parses_numeric_ipv4_and_ipv6_ports(self) -> None:
        tcp = ("  0: 0100007F:2CAA 00000000:0000 0A",)
        tcp6 = ("  0: 00000000000000000000000001000000:2CAA "
                "00000000000000000000000000000000:0000 0A",)

        self.assertTrue(_private_listen(tcp, tcp6))
        self.assertFalse(_private_listen(("  0: 0100007F:2C9A 00000000:0000 0A",), ()))

    def test_public_same_port_listener_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "public listener"):
            _private_listen(("  0: 00000000:2CAA 00000000:0000 0A",), ())

    def test_immutable_allows_root_owner_write_but_rejects_group_write_and_symlinks(self) -> None:
        paths = ("/fixed/bin/launcher", "/fixed/bin", "/fixed", "/")

        def root_owned(mode: int):
            return lambda _path: SimpleNamespace(st_uid=0, st_gid=0, st_mode=mode)

        _assert_immutable("/fixed/bin/launcher", lstat=root_owned(stat.S_IFREG | stat.S_IWUSR))
        with self.assertRaisesRegex(AssertionError, "mutable path"):
            _assert_immutable("/fixed/bin/launcher",
                              lstat=root_owned(stat.S_IFDIR | stat.S_IWGRP))
        with self.assertRaisesRegex(AssertionError, "symlink"):
            _assert_immutable("/fixed/bin/launcher",
                              lstat=root_owned(stat.S_IFLNK | stat.S_IWUSR))
        self.assertEqual(paths[-1], "/")


if __name__ == "__main__":
    unittest.main()
