// glx_context.cpp
#include <X11/Xlib.h>
#include <GL/glx.h>
#include <cstdio>
#include <cstdlib>

// g++ glx_context.cpp -o glx_context -lX11 -lGL


using PFNGLXCREATECONTEXTATTRIBSARBPROC =
    GLXContext(*)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

int main() {
    Display* dpy = XOpenDisplay(nullptr);
    if (!dpy) { std::printf("XOpenDisplay failed\n"); return 1; }

    static int fbAttribs[] = {
        GLX_X_RENDERABLE, True,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
        GLX_RED_SIZE,   8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE,  8,
        GLX_ALPHA_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        GLX_STENCIL_SIZE, 8,
        GLX_DOUBLEBUFFER, True,
        None
    };

    int ncfg = 0;
    GLXFBConfig* cfgs = glXChooseFBConfig(dpy, DefaultScreen(dpy), fbAttribs, &ncfg);
    if (!cfgs || ncfg == 0) { std::printf("glXChooseFBConfig failed\n"); return 1; }
    GLXFBConfig cfg = cfgs[0];

    XVisualInfo* vi = glXGetVisualFromFBConfig(dpy, cfg);
    if (!vi) { std::printf("glXGetVisualFromFBConfig failed\n"); return 1; }

    Colormap cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);

    XSetWindowAttributes swa{};
    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;

    Window win = XCreateWindow(
        dpy, RootWindow(dpy, vi->screen),
        0, 0, 800, 600, 0,
        vi->depth, InputOutput, vi->visual,
        CWColormap | CWEventMask, &swa
    );
    XStoreName(dpy, win, "Raw GLX Context");
    XMapWindow(dpy, win);

    auto glXCreateContextAttribsARB =
        (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddressARB(
            (const GLubyte*)"glXCreateContextAttribsARB"
        );
    if (!glXCreateContextAttribsARB) {
        std::printf("glXCreateContextAttribsARB not available\n");
        return 1;
    }

    int ctxAttribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB, 3,
        GLX_CONTEXT_PROFILE_MASK_ARB,  GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };

    GLXContext ctx = glXCreateContextAttribsARB(dpy, cfg, nullptr, True, ctxAttribs);
    if (!ctx) { std::printf("glXCreateContextAttribsARB failed\n"); return 1; }

    glXMakeCurrent(dpy, win, ctx);

    // Minimal event loop
    bool running = true;
    while (running) {
        while (XPending(dpy)) {
            XEvent e;
            XNextEvent(dpy, &e);
            if (e.type == DestroyNotify) running = false;
            if (e.type == KeyPress) running = false;
        }
        glXSwapBuffers(dpy, win);
    }

    glXMakeCurrent(dpy, None, nullptr);
    glXDestroyContext(dpy, ctx);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    return 0;
}
