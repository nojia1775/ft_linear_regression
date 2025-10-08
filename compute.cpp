#include "ARNetwork/neural_network/include/ARNetwork.hpp"

int	main(int argc, char **argv)
{
	// if (argc != 3)
	// {
	// 	std::cerr << "Error: ./compute <mileage> <file.json>" << std::endl;
	// 	return 1;
	// }
	(void)argc;
	(void)argv;
	try
	{
		std::vector<double> test({240000,
		139800,
		150500,
		185530,
		176000,
		114800,
		166800,
		89000,
		144500,
		84000,
		82029,
		63060,
		74000,
		97500,
		67000,
		76025,
		48235,
		93000,
		60949,
		65674,
		54000,
		68500,
		22899,
		61789,});
		ARNetwork arn("network.json");
		for (const auto& coef : test)
			std::cout << coef << ',' << arn.feed_forward({coef}, identity, identity)[0] * 8290 << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}