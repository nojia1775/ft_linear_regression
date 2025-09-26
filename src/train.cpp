#include "../include/ft_linear_regression.hpp"

int	main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "Error: wrong number of arguments" << std::endl;
		return 1;
	}
	try
	{
		Lr lr;
		lr.train(argv[1]);
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}