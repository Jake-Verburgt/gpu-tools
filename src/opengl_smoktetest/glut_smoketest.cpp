#include <GL/glut.h>
#include <time.h>


// g++ glut_smoketest.cpp -o test -lglut -lGL -lGLU
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("GLUT Context Test");

    glutDisplayFunc(display);
    glutMainLoop();
}