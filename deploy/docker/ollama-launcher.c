#define _GNU_SOURCE
#include <errno.h>
#include <grp.h>
#include <pwd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int trusted_regular(const char *path) {
    struct stat info;
    if (lstat(path, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != 0 ||
        (info.st_mode & (S_IWGRP | S_IWOTH)) != 0) return 0;
    return 1;
}

/* Fixed, argument-free privilege boundary: only the image's llm caller may
 * enter, and the only permitted result is the named ollama service identity. */
int main(int argc, char **argv) {
    struct passwd *entry;
    uid_t caller_uid;
    (void)argv;
    entry = getpwnam("llm");
    if (entry == NULL) return 126;
    caller_uid = entry->pw_uid;
    if (argc != 1 || geteuid() != 0 || getuid() != caller_uid) return 126;
    /* Do not retain pointers returned by getpwnam: its storage is static. */
    if (getpwnam("ollama") == NULL) return 126;
    if (!trusted_regular("/opt/venv/bin/python") ||
        !trusted_regular("/opt/llm/services/llm/bootstrap/ollama_broker.py")) return 126;
    if (clearenv() != 0 || setenv("PATH", "/usr/bin:/bin", 1) != 0 ||
        setenv("HOME", "/var/empty", 1) != 0 || chdir("/opt/llm") != 0) return 126;
    /* The caller grants only stdin/stdout, never an inherited file capability. */
    if (close_range(3, ~0U, 0) != 0) return 126;
    char *const command[] = { (char *)"/opt/venv/bin/python", (char *)"-I",
        (char *)"/opt/llm/services/llm/bootstrap/ollama_broker.py", 0 };
    execv(command[0], command);
    return errno == ENOENT ? 127 : 126;
}
