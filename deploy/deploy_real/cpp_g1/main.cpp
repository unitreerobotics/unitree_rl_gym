#include <iostream>
#include "Controller.h"

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout << "g1_deploy_real [net_interface]" << std::endl;
		exit(1);
	}
	Controller controller(argv[1]);
	controller.zero_torque_state();
	controller.move_to_default_pos();
	controller.run();
	controller.damp();
	return 0;
}
