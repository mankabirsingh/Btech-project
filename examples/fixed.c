#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void safe_copy(const char *user_input) {
    char stack_buffer[64];
    if (user_input == NULL) {
        return;
    }
    // Safe bounded copy preventing buffer overflow
    strncpy(stack_buffer, user_input, sizeof(stack_buffer) - 1);
    stack_buffer[sizeof(stack_buffer) - 1] = '\0';
    printf("User input received safely: %s\n", stack_buffer);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        safe_copy(argv[1]);
    }
    return 0;
}
