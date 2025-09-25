#include "../include/ft_linear_regression.hpp"

int	main(int argc, char **argv)
{
	(void)argv;
	if (argc != 2)
	{
		std::cerr << "Wrong number of argument" << std::endl;
		return 1;
	}
	try
	{
		Lr lr;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}