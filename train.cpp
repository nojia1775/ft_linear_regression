#include "ARNetwork/neural_network/include/ARNetwork.hpp"
#include <string>

static bool	valid_line(const std::string& line, const size_t& pos)
{
	size_t comma = 0;
	if (pos == 0)
	{
		if (line != "km,price")
			return false;
		return true;
	}
	for (size_t i = 0 ; i < line.size() ; i++)
	{
		if (!std::isdigit(line[i]) && line[i] != ',')
			return false;
		else if (line[i] == ',' && comma)
			return false;
		else if (line[i] == ',' && !std::isdigit(line[i + 1]))
			return false;
		else if (line[i] == ',')
			comma++;
	}
	if (comma == 0)
		return false;
	return true;
}

static std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>	extract_data(const std::string& file)
{
	std::ifstream data(file);
	if (!data)
		throw Error("Error: couldn't open " + file);
	std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> set;
	std::string line;
	size_t count = 0;
	while (getline(data, line))
	{
		if (valid_line(line, count) == false)
		{
			data.close();
			throw Error("Error: " + file + " is corrupted");
		}
		if (count != 0)
		{
			size_t pos = line.find(',');
			set.first.push_back({std::atof(line.substr(0, pos).c_str())});
			set.second.push_back({std::atof(line.c_str() + pos + 1)});
		}
		count++;
	}
	data.close();
	return set;
}

static double	normalize_data(std::vector<std::vector<double>>& data)
{
	double max = 0;
	for (const auto& vector : data)
		for (const auto& coef : vector)
			max = coef > max ? coef : max;
	for (auto& vector : data)
		for (auto& coef : vector)
			coef /= max;
	return max;
}

int	main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cerr << "Error: ./train <dataset> <epochs>" << std::endl;
		return 1;
	}
	try
	{
		std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data = extract_data(argv[1]);
		normalize_data(data.first);
		normalize_data(data.second);
		ARNetwork arn(std::vector<size_t>({1, 1}));
		arn.randomize_bias(0, 0);
		arn.randomize_weights(0, 0);
		arn.set_learning_rate(0.1);
		std::vector<double> loss = arn.train(PairFunction("MSE"), PairFunction("identity"), PairFunction("identity"), arn.batching(data.first, data.first.size()), arn.batching(data.second, data.second.size()), std::atoi(argv[2]));
		arn.get_json("network.json");
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}