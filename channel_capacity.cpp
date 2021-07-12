// ConsoleApplication1.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <stdlib.h>
#include <string>
#include <math.h>
#include <fstream>
#include <iostream>
#include <ctime>
#include <sstream>

using namespace std;

// Function for computing the gradient at a given point
void gradient_computation(double ** current_Q, double ** grad_Qjk, double grad_pj[], double q_k[], double xi_arg[], double xi_output[], int U_arg, double lambda_arg[], double lambda_output[], double p_arg[], double p_output[], int M_arg, double ***Qs_arg, double **Q0_arg, int N_arg, double a_arg[], double b_arg)
{
	for (int j = 0; j < M_arg; j++)
	{
		for (int k = 0; k < N_arg; k++)
		{
			current_Q[j][k] = Q0_arg[j][k];
		}
	}

	for (int u = 0; u < U_arg; u++)
	{
		for (int j = 0; j < M_arg; j++)
		{
			for (int k = 0; k < N_arg; k++)
			{
				current_Q[j][k] += xi_arg[u] * Qs_arg[u][j][k];
			}
		}
	}

	// Computing the current posterior probabilities 

	for (int k = 0; k < N_arg; k++)
	{
		q_k[k] = 0;
		for (int l = 0; l < M_arg; l++)
		{
			q_k[k] += p_arg[l] * current_Q[l][k];
		}

		//cout << q_k[k] << " ";
	}

	for (int j = 0; j < M_arg; j++)
	{
		grad_pj[j] = 1;

		for (int k = 0; k < N_arg; k++)
		{
			grad_pj[j] += current_Q[j][k] * log(current_Q[j][k] / q_k[k]); // Gradient of AMIM w.r.t. p
			grad_Qjk[j][k] = p_arg[j] * log(current_Q[j][k] / q_k[k]);
		}

		grad_pj[j] -= lambda_arg[0] * a_arg[j];

		p_output[j] = -grad_pj[j];
	}

	// Final making of the gradient w.r.t. xi
	for (int u = 0; u < U_arg; u++)
	{
		xi_output[u] = 0;

		for (int j = 0; j < M_arg; j++)
		{
			for (int k = 0; k < N_arg; k++)
			{
				xi_output[u] += grad_Qjk[j][k] * Qs_arg[u][j][k]; // Gradient of AMIM w.r.t. Qjk times the constant Q
			}
		}
	}

	lambda_output[0] = b_arg;

	for (int j = 0; j < M_arg; j++)
	{
		lambda_output[0] -= a_arg[j] * p_arg[j];
	}

}

double capacity_computation(double ** current_Q, double q_k[], double xi_arg[], int U_arg, double p_arg[], int M_arg, double ***Qs_arg, double **Q0_arg, int N_arg)
{
	for (int j = 0; j < M_arg; j++)
	{
		for (int k = 0; k < N_arg; k++)
		{
			current_Q[j][k] = Q0_arg[j][k];
			//cout << current_Q[j][k] << "\n";
		}
	}

	for (int u = 0; u < U_arg; u++)
	{
		for (int j = 0; j < M_arg; j++)
		{
			for (int k = 0; k < N_arg; k++)
			{
				current_Q[j][k] += xi_arg[u] * Qs_arg[u][j][k];
			}
		}
	}

	// Computing the current posterior probabilities 

	for (int k = 0; k < N_arg; k++)
	{
		q_k[k] = 0;
		for (int l = 0; l < M_arg; l++)
		{
			q_k[k] += p_arg[l] * current_Q[l][k];
		}
	}

	double capacity = 0;

	for (int j = 0; j < M_arg; j++)
	{
		for (int k = 0; k < N_arg; k++)
		{
			capacity += p_arg[j] * current_Q[j][k] * log(current_Q[j][k] / q_k[k]);
		}
	}

	return capacity;
}

void prox_operator_ellipsoid_orthant(double xi_arg[], double at_xi_arg[], double xi_output[], int U_arg, double lambda_arg[], double at_lambda_arg[], double lambda_output[], double p_arg[], double at_p_arg[], double p_output[], int M_arg, double gama_arg[], double delta_arg, double Lambda_max_arg)
{
	// Projecting onto the box
	double norm_of_the_projected = 0;

	for (int u = 0; u < U_arg; u++)
	{
		if ((at_xi_arg[u] * gama_arg[0] - xi_arg[u]) / gama_arg[0] > 0)
		{
			norm_of_the_projected += pow((at_xi_arg[u] * gama_arg[0] - xi_arg[u]) / gama_arg[0], 2);
		}
	}

	norm_of_the_projected = pow(norm_of_the_projected, 0.5);

	//cout << "Norm of the projected: " << norm_of_the_projected << "\n";

	if (norm_of_the_projected > 0)
	{
		if (norm_of_the_projected <= 1)
		{
			for (int u = 0; u < U_arg; u++)
			{
				if ((at_xi_arg[u] * gama_arg[0] - xi_arg[u]) / gama_arg[0] >= 0)
				{
					xi_output[u] = (at_xi_arg[u] * gama_arg[0] - xi_arg[u]) / gama_arg[0];
				}
				else
				{
					xi_output[u] = 0;
				}
			}
		}
		else
		{
			for (int u = 0; u < U_arg; u++)
			{
				if ((at_xi_arg[u] * gama_arg[0] - xi_arg[u]) / gama_arg[0] >= 0)
				{
					xi_output[u] = (at_xi_arg[u] * gama_arg[0] - xi_arg[u]) / gama_arg[0] / norm_of_the_projected;
				}
				else
				{
					xi_output[u] = 0;
				}
			}
		}
	}
	else
	{
		for (int u = 0; u < U_arg; u++)
		{
			xi_output[u] = 0;
		}
	}

	// Projecting onto the simplex

	double sum_of_exps = 0;

	for (int j = 0; j < M_arg; j++)
	{
		sum_of_exps = sum_of_exps + exp(-(p_arg[j] - gama_arg[1] * (1 + log(at_p_arg[j] + delta_arg / M_arg))) / gama_arg[1]);
	}

	for (int j = 0; j < M_arg; j++)
	{
		p_output[j] = exp(-(p_arg[j] - gama_arg[1] * (1 + log(at_p_arg[j] + delta_arg / (double)M_arg))) / gama_arg[1]) / sum_of_exps;
	}

	if ((at_lambda_arg[0] * gama_arg[0] - lambda_arg[0]) / gama_arg[0] < 0)
	{
		lambda_output[0] = 0;
	}
	else
	{
		if ((at_lambda_arg[0] * gama_arg[0] - lambda_arg[0]) / gama_arg[0] < Lambda_max_arg)
		{
			lambda_output[0] = (at_lambda_arg[0] * gama_arg[0] - lambda_arg[0]) / gama_arg[0];
		}
		else
		{
			lambda_output[0] = Lambda_max_arg;
		}
	}

}

int main(){

	ifstream infile;
	infile.open("CPP_data.txt");

	int number_of_file, M, N, U;
	double tau;

	infile >> number_of_file;
	infile >> M;
	infile >> N;
	infile >> U;
	infile >> tau;

	double ** Q0;
	double *** Qs;

	// The matrix with nominal channel probabilities
	Q0 = new double*[M];

	for (int j = 0; j < M; j++)
	{
		Q0[j] = new double[N];
		for (int k = 0; k < N; k++)
		{
			infile >> Q0[j][k];
		}
	}

	// The array with perturbation channel probabilities
	Qs = new double **[U];

	for (int u = 0; u < U; u++)
	{
		Qs[u] = new double*[M];
		for (int j = 0; j < M; j++)
		{
			Qs[u][j] = new double[N];
			for (int k = 0; k < N; k++)
			{
				infile >> Qs[u][j][k];
				//cout << Qs[u][j][k] << " ";
			}

			//cout << "\n";
		}
		//cout << "\n";
	}

	// Defining the constants for the whole thing

	double Q_norm = 0;

	for (int u = 0; u < U; u++)
	{
		for (int j = 0; j < M; j++)
		{
			for (int k = 0; k < N; k++)
			{
				Q_norm += pow(Qs[u][j][k], 2);
			}
		}
	}

	Q_norm = pow(Q_norm, 0.5);

	double alpha[2];
	alpha[0] = 1;
	alpha[1] = 0.5;

	double L[2][2];

	infile >> L[0][0];
	infile >> L[0][1];
	infile >> L[1][0];
	infile >> L[1][1];

	double * a;
	a = new double[M];
	for (int j = 0; j < M; j++)
	{
		infile >> a[j];
	}

	double b;
	infile >> b;

	double Lambda_max;
	infile >> Lambda_max;

	double delta = pow(10, -16);

	double Theta[2];
	Theta[0] = 2 + pow(Lambda_max, 2) / 2;
	Theta[1] = 4 * log(M / delta);

	clock_t time_begin = clock();
	clock_t time_end;

	infile.close();

	double M_array[2][2];

	M_array[0][0] = L[0][0] * Theta[0] / alpha[0];
	M_array[1][1] = L[1][1] * Theta[1] / alpha[1];
	M_array[0][1] = L[0][1] * pow(Theta[0] * Theta[1] / alpha[0] / alpha[1], 0.5);
	M_array[1][0] = M_array[0][1];

	double sigma[2];

	sigma[0] = (double)(M_array[0][0] + M_array[0][1]) / (double)(M_array[0][0] + M_array[0][1] + M_array[1][0] + M_array[1][1]);
	sigma[1] = (double)(M_array[1][0] + M_array[1][1]) / (double)(M_array[0][0] + M_array[0][1] + M_array[1][0] + M_array[1][1]);

	double gama[2];

	gama[0] = (double)sigma[0] / (double)Theta[0];
	gama[1] = (double)sigma[1] / (double)Theta[1];

	double a_tilde = 1;
	double Theta_tilde = 1;

	double L_tilde = 0;

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			L_tilde += L[i][j] * pow((double)Theta[i] * (double)Theta[j] / (double)alpha[i] / (double)alpha[j], 0.5);
		}
	}

	double gama_tilde = (double)a_tilde / (double)L_tilde / (double)pow(2, 0.5);


	// Initialization of the variables

	double * xi;
	xi = new double[U];
	double lambda[1];
	double * p;
	p = new double[M];
	double *to_project_xi;
	to_project_xi = new double[U];
	double to_project_lambda[1];
	double * to_project_p;
	to_project_p = new double[M];
	double * projected_w_xi;
	projected_w_xi = new double[U];
	double projected_w_lambda[1];
	double * projected_w_p;
	projected_w_p = new double[M];
	double * w_xi;
	w_xi = new double[U];
	double w_lambda[1];
	double * w_p;
	w_p = new double[M];
	double * w_xi_previous;
	w_xi_previous = new double[U];
	double w_lambda_previous[1];
	double * w_p_previous;
	w_p_previous = new double[M];
	double *p_final;
	p_final = new double[M];
	double lambda_final[1];
	double * xi_final;
	xi_final = new double[U];

	// Creating the matrices for the intermediate gradient values etc.

	double ** current_Q_global;
	current_Q_global = new double *[M];
	for (int j = 0; j < M; j++)
	{
		current_Q_global[j] = new double[N];
	}

	double ** grad_Qjk_global;
	grad_Qjk_global = new double *[M];
	for (int j = 0; j < M; j++)
	{
		grad_Qjk_global[j] = new double[N];
	}
	double * grad_pj_global;
	grad_pj_global = new double[M];

	// Computing the current posterior probabilities 
	double * q_k_global;
	q_k_global = new double[N];

	// cout << M;

	for (int j = 0; j < M; j++)
	{
		p[j] = 1 / (double)M;
	}

	for (int u = 0; u < U; u++)
	{
		xi[u] = 0.0;
	}

	lambda[0] = (double)0;

	cout << Theta_tilde << "\n";
	cout << gama_tilde << "\n";

	double sum_gamma = 0;
	int i_count_total = 0;
	int count_up = 0;
	double current_capacity = 0;
	int max_gama_reached = 0;

	ofstream outfile;
	ostringstream o_filename;
	o_filename << "saddle_output" << number_of_file << ".txt";
	outfile.open(o_filename.str().c_str());

	// The outer loop starts here
	for (int i_out = 0; i_out < pow(10, 6); i_out++)
	{
		// Optimality check

		gradient_computation(current_Q_global, grad_Qjk_global, grad_pj_global, q_k_global, xi, to_project_xi, U, lambda, to_project_lambda, p, to_project_p, M, Qs, Q0, N, a, b);
		prox_operator(to_project_xi, xi, projected_w_xi, U, to_project_lambda, lambda, projected_w_lambda, to_project_p, p, projected_w_p, M, gama, delta, Lambda_max);

		double optimality_condition = 0;

		for (int j = 0; j < M; j++)
		{
			optimality_condition += fabs(projected_w_p[j] - p[j]);
		}

		for (int u = 0; u < U; u++)
		{
			optimality_condition += fabs(projected_w_xi[u] - xi[u]);
		}

		optimality_condition += fabs(projected_w_lambda[0] - lambda[0]);

		if (optimality_condition < pow(10, -17))
		{
			cout << "Brejki brejki";
			break;
		}

		// Now the inner loop iteration starts

		for (int u = 0; u < U; u++)
		{
			w_xi[u] = xi[u];
		}

		for (int j = 0; j < M; j++)
		{
			w_p[j] = p[j];
		}

		w_lambda[0] = lambda[0];

		double inner_condition = 1;
		int inner_count = 1;

		while (inner_condition > pow(10, -17))
		{
			for (int u = 0; u < U; u++)
			{
				w_xi_previous[u] = w_xi[u];
			}

			for (int j = 0; j < M; j++)
			{
				w_p_previous[j] = w_p[j];
			}

			w_lambda_previous[0] = w_lambda[0];

			gradient_computation(current_Q_global, grad_Qjk_global, grad_pj_global, q_k_global, w_xi_previous, to_project_xi, U, w_lambda_previous, to_project_lambda, w_p_previous, to_project_p, M, Qs, Q0, N, a, b);

			for (int u = 0; u < U; u++)
			{
				to_project_xi[u] = to_project_xi[u] * gama_tilde;
			}

			for (int j = 0; j < M; j++)
			{
				to_project_p[j] = to_project_p[j] * gama_tilde;
			}

			to_project_lambda[0] = to_project_lambda[0] * gama_tilde;

			prox_operator(to_project_xi, xi, projected_w_xi, U, to_project_lambda, lambda, projected_w_lambda, to_project_p, p, projected_w_p, M, gama, delta, Lambda_max);

			inner_condition = 0;

			for (int u = 0; u < U; u++)
			{
				inner_condition += to_project_xi[u] * (w_xi[u] - projected_w_xi[u]) + gama[0] / (double)2 * pow(xi[u], 2) + (projected_w_xi[u] - xi[u]) * gama[0] * xi[u] - gama[0] / 2 * pow(projected_w_xi[u], 2);
			}

			for (int j = 0; j < M; j++)
			{
				inner_condition += to_project_p[j] * (w_p[j] - projected_w_p[j]) + gama[1] * (p[j] + delta / (double)M) * log(p[j] + delta / (double)M) + (projected_w_p[j] - p[j]) * gama[1] * (1 + log(p[j] + delta / (double)M)) - gama[1] * (projected_w_p[j] + delta / (double)M) * log(projected_w_p[j] + delta / (double)M);
			}

			inner_condition += to_project_lambda[0] * (w_lambda[0] - projected_w_lambda[0]) + gama[0] / (double)2 * pow(lambda[0], 2) + (projected_w_lambda[0] - lambda[0]) * gama[0] * lambda[0] - gama[0] / 2 * pow(projected_w_lambda[0], 2);

			if ((inner_count > 2) && (inner_condition > pow(10, -17)))
			{
				gama_tilde = gama_tilde / 1.6;
				max_gama_reached = 1;
				for (int u = 0; u < U; u++)
				{
					w_xi[u] = xi[u];
				}

				for (int j = 0; j < M; j++)
				{
					w_p[j] = p[j];
				}

				w_lambda[0] = lambda[0];

				inner_count = 1;
			}
			else
			{
				for (int u = 0; u < U; u++)
				{
					w_xi[u] = projected_w_xi[u];
				}

				for (int j = 0; j < M; j++)
				{
					w_p[j] = projected_w_p[j];
				}

				w_lambda[0] = projected_w_lambda[0];

				inner_count++;
			}

			//cout << "Inner count: " << inner_count <<  "\n";
		}

		// End of inner loop

		for (int u = 0; u < U; u++)
		{
			xi[u] = projected_w_xi[u];
		}

		for (int j = 0; j < M; j++)
		{
			p[j] = projected_w_p[j];
		}

		lambda[0] = projected_w_lambda[0];

		// Updating the final value

		for (int u = 0; u < U; u++)
		{
			xi_final[u] = (sum_gamma / (sum_gamma + gama_tilde)) * xi_final[u] + (gama_tilde / (sum_gamma + gama_tilde)) * w_xi_previous[u];
		}

		for (int j = 0; j < M; j++)
		{
			p_final[j] = (sum_gamma / (sum_gamma + gama_tilde)) * p_final[j] + (gama_tilde / (sum_gamma + gama_tilde)) * w_p_previous[j];
		}

		lambda_final[0] = (sum_gamma / (sum_gamma + gama_tilde)) * lambda_final[0] + (gama_tilde / (sum_gamma + gama_tilde)) * w_lambda_previous[0];

		sum_gamma += gama_tilde;

		gama_tilde = gama_tilde * 1.5;

		// Computing the value of the constraint:

		double cost_constraint = -b;

		for (int j = 0; j < M; j++)
		{
			cost_constraint += a[j] * p_final[j];
		}

		//cout << "Outer iteration: " << i_count_total << "\n";

		i_count_total++;

		if ((i_out % 1000) == 0)
		{
			cout << "Precision: " << Theta_tilde / sum_gamma << "\n";
			//cout << "Constraint value: " << cost_constraint << "\n";
			cout << "Capacity: " << capacity_computation(current_Q_global, q_k_global, xi_final, U, p_final, M, Qs, Q0, N) << "\n";
			//cout << gama_tilde << "\n";
		}

		if (Theta_tilde / sum_gamma < (1 * pow(10, -2)))
		{
			cout << "Brejki brejki precision met\n";
			break;
		}

	}

	//cout << i_count_total << "\n";

	for (int j = 0; j < M; j++)
	{
		cout << p_final[j] << " ";
	}
	cout << "\n";

	//outfile << xi[0] << "\n";

	outfile << "\r\nIterations: " << i_count_total << "\r\n";
	outfile << "\r\nXi final: ";
	for (int u = 0; u < U; u++)
	{
		outfile << xi_final[u] << "\r\n";
	}

	outfile << "\r\nCapacity: " << capacity_computation(current_Q_global, q_k_global, xi_final, U, p_final, M, Qs, Q0, N) << "\n";
	outfile << "\r\nPosterior guarantee: " << Theta_tilde / sum_gamma << "\r\n";

	//cout << count_up << "\n";
	time_end = clock() - time_begin;
	cout << time_end;
	outfile.close();

	return 0;
}

