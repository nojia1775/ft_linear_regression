#pragma once

#include <iostream>
#include <fstream>

class	Lr
{
	private:
		float		_weight;
		float		_bias;
		float		_learning_rate;

	public:
				Lr(void);
				~Lr(void) {}

		float		train(const std::ifstream& file);
		float		compute(const float& miles) const;

		const float&	getWeight(void) const { return _weight; }
		const float&	getBias(void) const { return _bias; }
		const float&	getLearningRate(void) const { return _learning_rate; }

		void		setWeight(const float& weight) { _weight = weight; }
		void		setBias(const float& bias) { _bias = bias; }
		void		setLearningRate(const float& learning_rate) { _learning_rate = learning_rate; }
};

class	Error : public std::exception
{
	private:
		std::string	_error;
	
	public:
				Error(const std::string& error) : _error(error) {}
				~Error(void) throw() {}

		const char	*what(void) const throw() { return _error.c_str(); }

};