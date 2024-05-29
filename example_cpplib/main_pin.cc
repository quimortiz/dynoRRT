// basic example using the collision manager
//
//
//
#include "dynoRRT/pin_col_manager.h"
#include <iostream>

using namespace dynorrt;

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt  <path_to_base_dir>" << std::endl;
    return 1;
  }

  std::string base_path(argv[1]);

  Collision_manager_pinocchio coll_manager;

  // see tests for meaningul and tested examples
  //
}
