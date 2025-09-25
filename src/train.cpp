#include "../include/ft_linear_regression.hpp"

int	main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "Wrong number of arguments" << std::endl;
		return 1;
	}
	std::ifstream file(argv[1], std::ios::in);
	if (!file)
		return 2;
	file.close();
	return 0;
}