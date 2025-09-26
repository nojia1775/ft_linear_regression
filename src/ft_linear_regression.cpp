#include "../include/ft_linear_regression.hpp"

static bool	valid_line(const std::string& line, const bool& positive)
{
	size_t dot = 0;
	for (size_t i = 0 ; i < line.size() ; i++)
	{
		if (!std::isdigit(line[i]) && line[i] != '.' && line[i] != '-')
			return false;
		else if (line[i] == '-' && (i != 0 || positive))
			return false;
		else if (line[i] == '.' && dot)
			return false;
		else if (line[i] == '.' && !std::isdigit(line[i + 1]))
			return false;
		else if (line[i] == '.')
			dot++;
	}
	return true;
}

static void	no_file(Lr& lr)
{
	std::ofstream newfile(".log.txt");
	if (!newfile)
		throw Error("Error: can't open file");
	newfile << "0\n0\n0.1";
	lr.setWeight(0);
	lr.setBias(0);
	lr.setLearningRate(0.1);
	newfile.close();
}

Lr::Lr(void) : _weight(0), _bias(0), _learning_rate(0.1)
{
	std::fstream file(".log.txt", std::ios::in);
	if (!file)
	{
		no_file(*this);
		return;
	}
	else
	{
		std::string line;
		size_t count = 0;
		while (std::getline(file, line))
		{
			if (count > 2)
			{
				no_file(*this);
				file.close();
				return;
			}
			if (!valid_line(line, count == 2 ? true : false))
			{
				no_file(*this);
				file.close();
				return;
			}
			switch (count)
			{
				case 0:
					_weight = std::atof(line.c_str());
					break;
				case 1:
					_bias = std::atof(line.c_str());
					break;
				case 2:
					_learning_rate = std::atof(line.c_str());
					break;
				default:
					break;
			}
			count++;
		}
		file.close();
		if (count != 3)
			no_file(*this);
	}
	std::cout << _weight << " " << _bias << " " << _learning_rate << std::endl;
}

float	Lr::train(const std::string& csv)
{
	std::ifstream file(csv);
	if (!file)
		throw Error("Error: can't open csv");
	std::map<float, float> datas;
	std::string line;
	size_t count = 0;
	while (std::getline(file, line))
	{
		if (!valid_line(line, true))
			throw Error("Error: csv is corrupted");
		if (count != 0)
		{
			size_t i = 0;
			while (line[i] != ',')
				i++;
			datas[std::atof(line.c_str())] = std::atof(line.c_str() + i + 1);
		}	
		count++;
	}
	file.close();
	return 1;
}