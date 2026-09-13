#define main b553_original_component_main
#include "x86_core.cc"
#undef main

int main(int argc, char** argv) {
  int iterations = 200000;
  int repeats = 10;
  int selected_p = 0;
  if (argc > 1) iterations = std::stoi(argv[1]);
  if (argc > 2) repeats = std::stoi(argv[2]);
  if (argc > 3) selected_p = std::stoi(argv[3]);



  if (selected_p == 0 || selected_p == 1)
    RunShape(1, 80, 16, iterations, repeats);
  if (selected_p == 0 || selected_p == 2)
    RunShape(2, 112, 8, iterations, repeats);
  if (selected_p == 0 || selected_p == 4)
    RunShape(4, 160, 4, iterations, repeats);
  if (selected_p == 0 || selected_p == 8)
    RunShape(8, 224, 2, iterations, repeats);
  if (selected_p == 0 || selected_p == 16)
    RunShape(16, 320, 1, iterations, repeats);
  return g_sink == 12345.0F ? 1 : 0;
}
