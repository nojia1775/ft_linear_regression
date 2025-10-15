#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include "../../linear_algebra/include/LinearAlgebra.hpp"

class	IActivation
{
	public:
		virtual			~IActivation(void) {}
		virtual std::string	name(void) const = 0;
		virtual	double		activate(const double& x) const { (void)x; throw Error("Error: function is not scalar based"); }
		virtual	double		derive(const double& x) const { (void)x; throw Error("Error: function is not scalar based"); }
		virtual	Vector<double>	activate(const Vector<double>& vector) { (void)vector; throw Error("Error: function is not vector-based"); }
		virtual	Matrix<double>	derive(const Vector<double>& vector) { (void)vector; throw Error("Error: function is not vector-based"); }
};

class	ReLU : public IActivation
{
	std::string	name(void) const { return "relu"; }
	double		activate(const double& x) const { return x <= 0 ? 0 : x; }
	double		derive(const double& x) const { return x <= 0 ? 0 : 1; }
};

class	Sigmoid : public IActivation
{
	std::string	name(void) const { return "sigmoid"; }
	double		activate(const double& x) const { return 1 / (1 + exp(-x)); }
	double		derive(const double& x) const { return activate(x) * (1 - activate(x)); }
};

class	TanH : public IActivation
{
	std::string	name(void) const { return "tanh"; }
	double		activate(const double& x) const { return std::tanh(x); }
	double		derive(const double& x) const { return 1 - std::tanh(x) * std::tanh(x); }
};

class	LeakyReLU : public IActivation
{
	std::string	name(void) const { return "leakyrelu"; }
	double		activate(const double& x) const { return x <= 0 ? x * 0.01 : x; }
	double		derive(const double& x) const { return x <= 0 ? 0.01 : 1; }
};

class	Identity : public IActivation
{
	std::string	name(void) const { return "identity"; }
	double		activate(const double& x) const { return x; }
	double		derive(const double& x) const { return x >= 0 ? 1 : -1; }
};

class	SoftMax : public IActivation
{
	std::string	name(void) const { return "softmax"; }
	Vector<double>	activate(const Vector<double>& x) const;
	Matrix<double>	derive(const Vector<double>& x) const;
};

class	ILoss
{
	public:
		virtual			~ILoss(void) {}
		virtual	std::string	name(void) const = 0;
		virtual double		activate(const Vector<double>& a, const Vector<double>& b) = 0;
		virtual Matrix<double>	derive(const Vector<double>& a, const Vector<double>& b) = 0;
};

class	MSE : public ILoss
{
	std::string	name(void) const { return "mse"; }
	double		activate(const Vector<double>& a, const Vector<double>& b) const;
	Matrix<double>	derive(const Vector<double>& a, const Vector<double>& b) const;
};

class	BCE : public ILoss
{
	std::string	name(void) const { return "bce"; }
	double		activate(const Vector<double>& a, const Vector<double>& b) const;
	Matrix<double>	derive(const Vector<double>& a, const Vector<double>& b) const;
};

class	ActivationFactory
{
	public:
		static std::unique_ptr<IActivation>	create(const std::string& function);
};

class	LossFactory
{
	public:
		static std::unique_ptr<ILoss>	create(const std::string& function);
};