// egl_x11_context.cpp
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <X11/Xlib.h>
#include <cstdio>

// g++ egl_x11_context.cpp -o egl_x11_context -lEGL -lX11 -lGL


int main() {
    Display* x_dpy = XOpenDisplay(nullptr);
    if (!x_dpy) { std::printf("XOpenDisplay failed\n"); return 1; }

    // EGL display tied to X11 display
    EGLDisplay egl_dpy = eglGetDisplay((EGLNativeDisplayType)x_dpy);
    if (egl_dpy == EGL_NO_DISPLAY) { std::printf("eglGetDisplay failed\n"); return 1; }

    EGLint major=0, minor=0;
    if (!eglInitialize(egl_dpy, &major, &minor)) { std::printf("eglInitialize failed\n"); return 1; }

    // Choose config
    EGLint cfgAttribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE
    };

    EGLConfig cfg;
    EGLint ncfg;
    if (!eglChooseConfig(egl_dpy, cfgAttribs, &cfg, 1, &ncfg) || ncfg < 1) {
        std::printf("eglChooseConfig failed\n"); return 1;
    }

    // Bind OpenGL API (not GLES)
    if (!eglBindAPI(EGL_OPENGL_API)) { std::printf("eglBindAPI OpenGL failed\n"); return 1; }

    // Create X11 window (minimal)
    Window root = DefaultRootWindow(x_dpy);
    Window win = XCreateSimpleWindow(x_dpy, root, 0, 0, 800, 600, 0, 0, 0);
    XMapWindow(x_dpy, win);
    XStoreName(x_dpy, win, "Raw EGL (X11) Context");

    EGLSurface surf = eglCreateWindowSurface(egl_dpy, cfg, (EGLNativeWindowType)win, nullptr);
    if (surf == EGL_NO_SURFACE) { std::printf("eglCreateWindowSurface failed\n"); return 1; }

    // Create context (OpenGL)
    EGLint ctxAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_NONE
    };
    EGLContext ctx = eglCreateContext(egl_dpy, cfg, EGL_NO_CONTEXT, ctxAttribs);
    if (ctx == EGL_NO_CONTEXT) { std::printf("eglCreateContext failed\n"); return 1; }

    if (!eglMakeCurrent(egl_dpy, surf, surf, ctx)) { std::printf("eglMakeCurrent failed\n"); return 1; }

    // loop-ish
    // for (int i = 0; i < 300; i++) {
    //     eglSwapBuffers(egl_dpy, surf);
    // }

   while (true) {
        eglSwapBuffers(egl_dpy, surf);
    }

    eglMakeCurrent(egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(egl_dpy, ctx);
    eglDestroySurface(egl_dpy, surf);
    eglTerminate(egl_dpy);
    XCloseDisplay(x_dpy);
    return 0;
}
