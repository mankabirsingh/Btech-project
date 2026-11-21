#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vulnerable_copy(const char *user_input) {
    char stack_buffer[64];
    // Unbounded string copy vulnerability: CWE-120 / CWE-119
    strcpy(stack_buffer, user_input);
    printf("User input received: %s\n", stack_buffer);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_copy(argv[1]);
    }
    return 0;
}
