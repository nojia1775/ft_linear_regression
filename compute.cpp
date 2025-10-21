#include "ARNetwork/neural_network/include/ARNetwork.hpp"

int	main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cerr << "Error: ./compute <mileage> <file.json>" << std::endl;
		return 1;
	}
	try
	{
		ARNetwork arn(argv[2]);
		std::cout << arn.feed_forward({std::atof(argv[1]) / 240000}, "identity", "identity")[0] * 8290 << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}