#ifndef CFD_H_
#define CFD_H_

/****************************************************************
*
*		CFD.h
*
*		Header file for
*		CFD Class
*
****************************************************************/

#include <iostream>
#include <algorithm>
#include <omp.h>
#include "glm\glm.hpp"	// Math library
using namespace glm;

typedef vec3 color;


class CFD {

public:
	// field types
	enum {SEMI_LAGRANGIAN, MODIFIED_MACCORMACK};
	// advection
	enum {DENSITY, VELOCITY, SOURCE, PRESSURE, COLOR};
	
	// constructor
	CFD();
	CFD(int _Nx, int _Ny, float _dc, const float *color_map);
	// destructor
	~CFD();

	// main simulation update method
	void sim_one_step(float time_step);

	// control methods
	void 	set_advecrtion(int TYPE);
	void 	set_color(int index, const color &c);
	void 	add_source(int index, float intensity);
	void 	add_obstruction(int index, float intensity);
	vec2 	add_gravity(const vec2 &gravity);

	int	increase_IOP_iteration();
	int  	decrease_IOP_iteration();
	int  	increase_GS_iteration();
	int  	decrease_GS_iteration();
	float 	increase_Kviscosity();
	float 	decrease_Kviscosity();
	float 	increase_Ktension();
	float 	decrease_Ktension();

	bool 	toogle_kinematicV();
	bool 	toogle_surfaceT();

	// get methods
	color 	get_field_color(int x, int y);
	color 	get_field_color(int index);
	int	get_field_obstruction(int x, int y);
	float 	get_pressure_value(int x, int y);
	float 	get_field_density(int x, int y);


private:

	// grid representation
	int   Nx, Ny;	// # of grids
	float dx, dy;	// cell size	** we assume dx = dy.

	// forces
	vec2 gravity;

	// constants
	float k_kinematicViscosity;
	float k_surfaceTension;

	// controls
	int  advection_type;
	int  it_gaussSeidel;	// Gauess - Seidel (smoothing pressure)
	int  it_IOP;			// Iterative Orthogonal Projection (Incompresibility)
	bool use_kinematicV;	// using kinematic viscosity
	bool use_surfaceT;		// using surface tension

	// field maps
	float *field_density;
	float *field_velocity;
	float *field_source;
	float *field_force;
	float *field_pressure;
	float *field_divergence;
	float *field_obstruction;
	float *field_color;
	float *field_kinematicV;
	float *field_surfaceT;
	float *field_MMCCharacteristicMap;

	// simulate methods
	void sim_density(float t);
	void sim_velocity(float t);
	void apply_source();
	void apply_force(float t);
	void gen_divergence();
	void sim_incompressibility();
	void sim_boundary();
	void sim_color(float t);
	void sim_kinematicV(float t);
	void sim_surfaceT(float t);
	// advection methods
	void semi_lagrangian(float x, float y, float &semi_x, float &semi_y, float vx, float vy, float t);
	void gen_MMCmap(float t);
	template<class T>
	void advection(int x, int y, float t, T &v, int FIELD_TYPE);

	// helper methods
	void clamp(int &val, int min, int max);
	void clamp(float &val, float min, float max);
	void init_field(float *field, int x, int y, float value);
	void clone_color_field(int x, int y, const float *source_field, float *target_field);
	vec3 bilinear_interpolate(float x, float y, int FIELD_TYPE);
};

#endif
