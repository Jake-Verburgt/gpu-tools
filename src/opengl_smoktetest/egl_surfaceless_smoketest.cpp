// egl_surfaceless.cpp
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cstdio>

// g++ egl_surfaceless.cpp -o egl_surfaceless -lEGL -lGL


int main() {
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) { std::printf("eglGetDisplay failed\n"); return 1; }

    EGLint major=0, minor=0;
    if (!eglInitialize(dpy, &major, &minor)) { std::printf("eglInitialize failed\n"); return 1; }

    EGLint cfgAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,            // pbuffer is the portable “no-window” surface
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_NONE
    };

    EGLConfig cfg;
    EGLint ncfg;
    if (!eglChooseConfig(dpy, cfgAttribs, &cfg, 1, &ncfg) || ncfg < 1) {
        std::printf("eglChooseConfig failed\n"); return 1;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) { std::printf("eglBindAPI failed\n"); return 1; }

    EGLint pbAttribs[] = {
        EGL_WIDTH,  64,
        EGL_HEIGHT, 64,
        EGL_NONE
    };
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pbAttribs);
    if (surf == EGL_NO_SURFACE) { std::printf("eglCreatePbufferSurface failed\n"); return 1; }

    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, nullptr);
    if (ctx == EGL_NO_CONTEXT) { std::printf("eglCreateContext failed\n"); return 1; }

    if (!eglMakeCurrent(dpy, surf, surf, ctx)) { std::printf("eglMakeCurrent failed\n"); return 1; }

    // You now have a headless-ish OpenGL context (backed by a pbuffer)
    // Do GL work here (usually with GL loader like glad), then read back pixels if needed.

    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglDestroySurface(dpy, surf);
    eglTerminate(dpy);
    return 0;
}
