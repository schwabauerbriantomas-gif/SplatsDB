#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int trace_fd = -1;
static void trace(const char *fmt, ...) {
    if (trace_fd < 0) {
        const char *p = getenv("F06B_TRACE");
        if (!p || !*p) return;
        trace_fd = (int)syscall(SYS_openat, AT_FDCWD, p, O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (trace_fd < 0) return;
    }
    char buf[8192];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) syscall(SYS_write, trace_fd, buf, (size_t)n);
}

static int wal_fd(int fd, char *path, size_t cap) {
    char link[64];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, path, cap - 1);
    if (n <= 0) { path[0] = 0; return 0; }
    path[n] = 0;
    size_t len = (size_t)n;
    const char *suffix = "/wal/store.wal";
    return len >= strlen(suffix) && !strcmp(path + len - strlen(suffix), suffix);
}

static int armed(void) {
    const char *p = getenv("F06B_ARM_FILE");
    return p && *p && access(p, F_OK) == 0;
}

ssize_t write(int fd, const void *buf, size_t count) {
    char path[PATH_MAX];
    int is_wal = wal_fd(fd, path, sizeof(path));
    if (is_wal && armed()) {
        const char *fail = getenv("F06B_FAIL_WRITE");
        if (fail && *fail) {
            int e = !strcmp(fail, "EIO") ? EIO : ENOSPC;
            size_t partial = getenv("F06B_PARTIAL_WRITE") ? count / 2 : 0;
            ssize_t rc = partial ? (ssize_t)syscall(SYS_write, fd, buf, partial) : 0;
            trace("WAL_WRITE path=%s requested=%zu partial=%zd fail=%s errno=%d\n", path, count, rc, fail, e);
            errno = e;
            return -1;
        }
    }
    ssize_t rc = (ssize_t)syscall(SYS_write, fd, buf, count);
    if (is_wal) trace("WAL_WRITE path=%s requested=%zu rc=%zd errno=%d\n", path, count, rc, rc < 0 ? errno : 0);
    return rc;
}

int fsync(int fd) {
    char path[PATH_MAX];
    int is_wal = wal_fd(fd, path, sizeof(path));
    if (is_wal && armed() && getenv("F06B_FAIL_FSYNC")) {
        trace("WAL_FSYNC path=%s injected errno=%d\n", path, EIO);
        errno = EIO;
        return -1;
    }
    int rc = (int)syscall(SYS_fsync, fd);
    if (is_wal) trace("WAL_FSYNC path=%s rc=%d errno=%d\n", path, rc, rc ? errno : 0);
    return rc;
}

int rename(const char *old, const char *new) {
    int rc = (int)syscall(SYS_rename, old, new);
    trace("RENAME old=%s new=%s rc=%d errno=%d\n", old, new, rc, rc ? errno : 0);
    return rc;
}
