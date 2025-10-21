#include "ARNetwork/neural_network/include/ARNetwork.hpp"

int	main(int argc, char **argv)
{
	if (argc != 5)
	{
		std::cerr << "Error: ./compute <mileage> <file.json> <layer_function> <output_function>" << std::endl;
		return 1;
	}
	try
	{
		double mileage = std::atof(argv[1]);
		if (mileage < 0)
			throw Error("Error: mileage cannot be negative");
		ARNetwork arn(argv[2]);
		double price = arn.feed_forward({std::atof(argv[1]) / 240000}, argv[3], argv[4])[0] * 8290;
		if (price < 0)
			price = 0;
		std::cout << "Mileage: " << mileage << "\nPrice: " << price << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}