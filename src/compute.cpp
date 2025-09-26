#include "../include/ft_linear_regression.hpp"

int	main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "Error: wrong number of argument" << std::endl;
		return 1;
	}
	try
	{
		Lr lr;
		std::cout << argv[1] << " mile(s) is $" << lr.compute(std::atof(argv[1])) << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}