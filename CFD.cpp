/*******************************************************************
*
*		CFD.cpp
*		
*		Implementation of CFD Class
*
*		Note.
*		-EVERY GRID FIELD is stored in CFD Object.
*		-That includes the color map for display.
*		-Values(Density, Color...) can be added to CFD object		
*		 by pixel unit using add_ methods.
*
*******************************************************************/

#include "CFD.h"


//**************************************//
//		con/destructors		//
//**************************************//

CFD::CFD() {};
CFD::CFD(int _Nx, int _Ny, float _dc, const float *color_map) : Nx(_Nx), Ny(_Ny), dx(_dc), dy(_dc)
{
	// enable openMP nested parallelism
	omp_set_nested(1);

	// init. buoyancy
	gravity = vec2(0, 9.8 * 50);

	// create fields
	field_density			= new float[Nx * Ny];
	field_velocity			= new float[Nx * Ny * 2];
	field_source			= new float[Nx * Ny];
	field_force			= new float[Nx * Ny * 2];
	field_divergence		= new float[Nx * Ny];
	field_pressure			= new float[Nx * Ny];
	field_obstruction		= new float[Nx * Ny];
	field_color			= new float[Nx * Ny * 3];
	field_kinematicV		= new float[Nx * Ny * 2];
	field_surfaceT			= new float[Nx * Ny * 2];
	field_MMCCharacteristicMap	= new float[Nx * Ny * 2];

	// initialize fields
	init_field(field_density, Nx, Ny, 0);
	init_field(field_velocity, Nx, Ny * 2, 0);
	init_field(field_source, Nx, Ny, 0);
	init_field(field_force, Nx, Ny * 2, 0);
	init_field(field_divergence, Nx, Ny, 0);
	init_field(field_pressure, Nx, Ny, 0);
	init_field(field_obstruction, Nx, Ny, 1);
	init_field(field_kinematicV, Nx, Ny, 2);
	init_field(field_MMCCharacteristicMap, Nx, Ny * 2, 0);
	clone_color_field(Nx, Ny, color_map, field_color);

	k_kinematicViscosity	= 0.01;
	k_surfaceTension	= exp(-2);
	it_gaussSeidel		= 5;
	it_IOP			= 5;

	advection_type	= SEMI_LAGRANGIAN;
	use_kinematicV	= false;
	use_surfaceT	= false;
};

// destructor
CFD::~CFD() {};




//**************************************//
//		THE MAIN simulation method		//
//**************************************//

// MAIN method called in each frame.
// simultates entire CFD due to input timestep.
void CFD::sim_one_step(float time_step) {
	
	// simulate CFD by time_step
	if (advection_type == MODIFIED_MACCORMACK) gen_MMCmap(time_step);

	sim_density(time_step);					// 1. move density
	sim_velocity(time_step);				// 2. move velocity
	sim_color(time_step);					// 3. move color
	if (use_kinematicV) sim_kinematicV(time_step);
	if (use_surfaceT)   sim_surfaceT(time_step);
	apply_source();						// 4. apply source
	apply_force(time_step);					// 5. apply force
	for (int i = 0; i < it_IOP; i++) {
		sim_incompressibility();			// 6. sim incompressibility
		sim_boundary();					// 7. sim boundary condition
	}
};




//***************************************//
//		field add methods	 //
//***************************************//

// update color field with input index and color
// Precondition: input index must be pixel point in
// XY Coordinate, with RGB value.
void CFD::set_color(int index, const color &c) {

	field_color[index + 0] = c.r;
	field_color[index + 1] = c.b;
	field_color[index + 2] = c.g;
};


// add source to source field due to index and intensity
// Precondition: input index must be pixel point in XY Coordinate
void CFD::add_source(int index, float intensity) {

	field_source[index] += intensity;
};


void CFD::add_obstruction(int index, float intensity) {
	
	field_obstruction[index] = intensity;
};


// Returns color of input XY position in color field
// generate index and pass to another get_field_color()
color CFD::get_field_color(int x, int y) { 
	
	int index = (y * Nx + x) * 3;
	
	return get_field_color(index);
};


// Precondition: input index must be pixel point in the XY Coordinate 
// Returns color of input XY position in color field
color CFD::get_field_color(int index) {

	float r = field_color[index + 0];
	float g = field_color[index + 1];
	float b = field_color[index + 2];

	return color(r, g, b);
};



float CFD::get_pressure_value(int x, int y) {
	return field_pressure[y * Nx + x];
};


int CFD::get_field_obstruction(int x, int y) {
	return field_obstruction[y * Nx + x];
};


// Returns density value of input XY position in color field
// Precondition: input index must be pixel point in the XY Coordinate 
float CFD::get_field_density(int x, int y) { 
	int index = (y * Nx + x);
	return field_density[index];
};


// Precondition: Must be only intesity, which not includes mass.
// Returns updated gravity value
vec2 CFD::add_gravity(const vec2 &g) {
	gravity += g;
	return gravity;
}

void CFD::set_advecrtion(int TYPE) 	{ advection_type = TYPE; };
int  CFD::increase_GS_iteration()	{ return ++it_gaussSeidel; };
int  CFD::decrease_GS_iteration()	{ return --it_gaussSeidel; };
int	 CFD::increase_IOP_iteration()	{ return ++it_IOP; };
int  CFD::decrease_IOP_iteration()	{ return --it_IOP; };
float  CFD::increase_Kviscosity()	{ return k_kinematicViscosity += 0.001; };
float  CFD::decrease_Kviscosity()	{ return k_kinematicViscosity -= 0.001; };
float  CFD::increase_Ktension()		{ return k_surfaceTension	  += 0.025; };
float  CFD::decrease_Ktension()		{ return k_surfaceTension	  -= 0.025; };
bool CFD::toogle_kinematicV()		{ return use_kinematicV = !use_kinematicV; };
bool CFD::toogle_surfaceT()		{ return use_surfaceT = !use_surfaceT; };

//***************************************//
//		local simulation methods		 //
//***************************************//

// Simulates density going over every grid points,
// applying advection due to velocity field.
void CFD::sim_density(float t) {

	for (int y = 0; y < Ny; y++) {
#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {
			vec2 updated_value;
			advection(x, y, t, updated_value, DENSITY);	

			field_density[y * Nx + x] = updated_value.x;
		}
	}
};


// Simulates velocity going over every grid points,
// applying advection due to velocity field.
void CFD::sim_velocity(float t) {
	
	// create copy version of velocity field
	float *temp_velocity_field = new float[Nx * Ny * 2];
	#pragma omp parallel for
	for (int i = 0; i < Nx*Ny * 2; i++) temp_velocity_field[i] = field_velocity[i];

	for (int y = 0; y < Ny; y++) {
	#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {

			vec2 updated_value;
			advection(x, y, t, updated_value, VELOCITY);

			temp_velocity_field[(y * Nx + x) * 2 + 0] = updated_value.x;	//x
			temp_velocity_field[(y * Nx + x) * 2 + 1] = updated_value.y;	//y
		}
	}

	// paste into original field
	
	float *temp_holder = field_velocity;
	field_velocity = temp_velocity_field;

	delete[] temp_holder;
};


// Simulates velocity going over every grid points,
// applying advection due to velocity field.
void CFD::sim_color(float t) {

	// create copy version of velocity field
	float *temp_color_field = new float[Nx * Ny * 3];
#pragma omp parallel for
	for (int i = 0; i < Nx*Ny * 3; i++) temp_color_field[i] = field_color[i];

	// move color by velocity
	for (int y = 0; y < Ny; y++) {
#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {

			// get new value
			vec3 updated_value;
			advection(x, y, t, updated_value, COLOR);
	
			// update density field
			temp_color_field[(y * Nx + x) * 3 + 0] = updated_value.r;	//r
			temp_color_field[(y * Nx + x) * 3 + 1] = updated_value.g;	//g
			temp_color_field[(y * Nx + x) * 3 + 2] = updated_value.b;	//b
		}
	}

	// paste into original field
	float *temp_holder = field_color;
	field_color = temp_color_field;

	delete[] temp_holder;
};


// simulate kinematic viscosity
// A physical phenominon where atoms create frictions.
void CFD::sim_kinematicV(float t) {

	for (int y = 1; y < Ny-1; y++) {
#pragma omp parallel for
		for (int x = 1; x < Nx-1; x++) {
			
			float kvx = (field_velocity[((y + 1) * Nx + x) * 2 + 0]
				+ field_velocity[((y - 1) * Nx + x) * 2 + 0]
				+ field_velocity[(y * Nx + x + 1) * 2 + 0]
				+ field_velocity[(y * Nx + x - 1) * 2 + 0]
				- field_velocity[(y * Nx + x) * 2 + 0]) * k_kinematicViscosity * t;

			float kvy = (field_velocity[((y + 1) * Nx + x) * 2 + 1]
				+ field_velocity[((y - 1) * Nx + x) * 2 + 1]
				+ field_velocity[(y * Nx + x + 1) * 2 + 1]
				+ field_velocity[(y * Nx + x - 1) * 2 + 1]
				- field_velocity[(y * Nx + x) * 2 + 1]) * k_kinematicViscosity * t;

			// update velocity 
			field_velocity[(y * Nx + x) * 2 + 0] = (field_velocity[(y * Nx + x) * 2 + 0] + kvx / (dx * dx)) * exp(-4 * k_kinematicViscosity * t);
			field_velocity[(y * Nx + x) * 2 + 1] = (field_velocity[(y * Nx + x) * 2 + 1] + kvy / (dy * dy)) * exp(-4 * k_kinematicViscosity * t);
		}
	}
};


// simulate surface tension
// Acts to eliminate strong curvature.
void CFD::sim_surfaceT(float t) {

	for (int y = 1; y < Ny - 1; y++) {
#pragma omp parallel for
		for (int x = 1; x < Nx - 1; x++) {
		
			// gradient values
			float gradx = (field_density[y * Nx + x + 1] - field_density[y * Nx + x - 1]) / (2 * dx);
			float grady	= (field_density[(y + 1) * Nx + x] - field_density[(y - 1) * Nx + x]) / (2 * dy);
			
			// mean values
			float mxx	= (field_density[y * Nx + x + 1] + field_density[y * Nx + x - 1] - 2 * field_density[y * Nx + x]) / (dx * dx);
			float myy	= (field_density[(y + 1) * Nx + x] + field_density[(y - 1) * Nx + x] - 2 * field_density[y * Nx + x]) / (dy * dy);
			float mxy	= (field_density[(y + 1) * Nx + x + 1] + field_density[(y - 1) * Nx + x - 1] - field_density[(y - 1) * Nx + x + 1] - field_density[(y + 1) * Nx + x - 1]) / (dx * dy * 4);
			float mag_grad = sqrt(gradx * gradx + grady * grady);

			float k_meanCurvature = (mxx + myy - (gradx * gradx) * mxx - (grady * grady) * myy + 2 * gradx * grady * mxy) / mag_grad;
			
			if (mag_grad >= 1e-10) {		// avoid dividing by zero
				field_force[(y * Nx + x) * 2 + 0] = k_meanCurvature * (gradx / mag_grad) * k_surfaceTension;
				field_force[(y * Nx + x) * 2 + 1] = k_meanCurvature * (grady / mag_grad) * k_surfaceTension;
			}
		}
	}
};


// Update density field due to current source field
// When it is updated, reset source to 0.
void CFD::apply_source() {
	
	#pragma omp parallel for
	for (int i = 0; i < Nx*Ny; i++) {

		field_density[i] += field_source[i];
		field_source[i] = 0;
	}
};


// Update velocity field due input force
// Precondition: force must be pure intensity with
//				 NO mass or density included.
void CFD::apply_force(float t) {

	for (int y = 0; y < Ny; y++) {
	#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {
			int index = (y * Nx + x) * 2;

			vec2 G(gravity.x * field_density[index / 2],		 // gravity force
				   gravity.y * field_density[index / 2]);

			field_velocity[index + 0] += (field_force[index + 0] + G.x) * t;
			field_velocity[index + 1] += (field_force[index + 1] + G.y) * t;
		}
	}

	init_field(field_force, Nx, Ny * 2, 0);
};


// Generates divergence field due to velocity field.
void CFD::gen_divergence() {
	
	for (int y = 0; y < Ny; y++) {
	#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {
			
			float vx, vX, vy, vY;

			// clamp
			if (x-1 < 0)	vx = 0;											//	------vY-----
			else			vx = field_velocity[(y * Nx + x - 1) * 2 + 0];	//	|			|
			if (x+1 >= Nx)	vX = 0;											//	|	 		|
			else			vX = field_velocity[(y * Nx + x + 1) * 2 + 0];	//	vx		   vX
			if (y-1 < 0)	vy = 0;											//	|	 		|
			else			vy = field_velocity[((y - 1) * Nx + x) * 2 + 1];//	|	 		|
			if (y+1 >= Ny)	vY = 0;											//	------vy-----
			else			vY = field_velocity[((y + 1) * Nx + x) * 2 + 1];

			float dvx = (vX - vx) / (2 * dx);			// delta velocity x
			float dvy = (vY - vy) / (2 * dy);			// delta velocity y

			field_divergence[y * Nx + x] = dvx + dvy;	// update divergence
		}
	}
}


// simulate incompressibility due to divergence and pressure
// As it iterates, the pressure smooths out in divergence (Gauss - Seidel)
// then update velocity field 
void CFD::sim_incompressibility() {
	
	gen_divergence();								// first, generate divergence field
												
	for (int i = 0; i < it_gaussSeidel; i++) {		// update pressure iteratively, until converges(Approx.) :: Gauss-Seidel Method
		for (int y = 0; y < Ny; y++) {
		///#pragma omp parallel for					// shouldn't use multi-thread in here! we are reading/writing in order
			for (int x = 0; x < Nx; x++) {

				float px, pX, py, pY;

				// clamp
				if (x-1 < 0)	px = 0;									//	------pY-----
				else			px = field_pressure[y * Nx + x - 1];	//	|			|
				if (x+1 >= Nx)	pX = 0;									//	|	 		|
				else			pX = field_pressure[y * Nx + x + 1];	//	px		   pX
				if (y-1 < 0)	py = 0;									//	|	 		|
				else			py = field_pressure[(y - 1) * Nx + x];	//	|	 		|
				if (y+1 >= Ny)	pY = 0;									//	------py-----
				else			pY = field_pressure[(y + 1) * Nx + x];

				field_pressure[y * Nx + x] = (1 / 4.f) * (px + pX + py + pY) - (1 / 4.f) * (field_divergence[y * Nx + x] * dx * dy);
			}
		}
	}

	for (int y = 0; y < Ny; y++) {			// update veclocity
	#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {

			float px, pX, py, pY;

			// clamp
			if (x-1 < 0)	px = 0;									//	------pY-----
			else			px = field_pressure[y * Nx + x - 1];	//	|			|
			if (x+1 >= Nx)	pX = 0;									//	|	 		|
			else			pX = field_pressure[y * Nx + x + 1];	//	px		   pX
			if (y-1 < 0)	py = 0;									//	|	 		|
			else			py = field_pressure[(y - 1) * Nx + x];	//	|	 		|
			if (y+1 >= Ny)	pY = 0;									//	------py-----
			else			pY = field_pressure[(y + 1) * Nx + x];

			field_velocity[(y * Nx + x) * 2 + 0] -= (pX - px) / (2 * dx);	// delta pressure x
			field_velocity[(y * Nx + x) * 2 + 1] -= (pY - py) / (2 * dy);	// delta pressure y
		}
	}
};


// simulate boundary condition
// based on obstruction map and window
void CFD::sim_boundary() {

	for (int y = 0; y < Ny; y++) {
	#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {
			int index = y * Nx + x;

			// HACKING window boundary... (Set first pixels to 0)
			if (y == 0 || y == Ny - 1 || x == 0 || x == Nx - 1) field_obstruction[index] = 0;

			field_velocity[index * 2 + 0] *= field_obstruction[index];	// x
			field_velocity[index * 2 + 1] *= field_obstruction[index];	// y
		}	
	}
};


// Generates new point by using semi-lagrangian scheme
// Precondition: force must be pure intensity with
//				 NO mass or density included.
// Postcondition: returned point is based on actual grid size,
//				  NOT index of the grid.
void CFD::semi_lagrangian(float x, float y, float &semi_x, float &semi_y, float vx, float vy, float t) {

	int index = ((y * Nx) + x) * 2;

	///// 1. get velocity of the grid point
	///float vx = field_velocity[index + 0];
	///float vy = field_velocity[index + 1];

	// 2. back trace
	semi_x = x * dx - vx * t;
	semi_y = y * dy - vy * t;
};


// Alternative advection scheme to establish
// Courant-Friedrichs-Lewy(CFL) condition.
void CFD::gen_MMCmap(float t) {

	for (int y = 0; y < Ny; y++) {
#pragma omp parallel for
		for (int x = 0; x < Nx; x++) {
			int index = (y * Nx + x) * 2;
			
			// forward characteristic map
			float forward_x, forward_y;
			semi_lagrangian(x, y, forward_x, forward_y, field_velocity[(y * Nx + x) * 2 + 0], field_velocity[(y * Nx + x) * 2 + 1], t);
			vec2 new_velocity = bilinear_interpolate(forward_x, forward_y, VELOCITY);

			// backward characteristic map
			float backward_x, backward_y;		
			semi_lagrangian(forward_x / dx, forward_y / dy, backward_x, backward_y, new_velocity.x, new_velocity.y, -1.f * t);
			
			// calculate error
			float errorX = 0.5f * (x * dx - backward_x);
			float errorY = 0.5f * (y * dy - backward_y);

			// init. MMC - Charateristic map
			field_MMCCharacteristicMap[index + 0] = forward_x + errorX;
			field_MMCCharacteristicMap[index + 1] = forward_y + errorY;
		}
	}
}


// Advection method for simulating density, velocity, color.
// Uses semi-lagrangian advection scheme.
template<class T>
void CFD::advection(int x, int y, float t, T &v, int FIELD_TYPE) {

	float new_x, new_y;

	if (advection_type == SEMI_LAGRANGIAN) {
		semi_lagrangian(x, y, new_x, new_y, field_velocity[(y * Nx + x) * 2 + 0], field_velocity[(y * Nx + x) * 2 + 1], t);
	}
	else if (advection_type == MODIFIED_MACCORMACK) {
		new_x = field_MMCCharacteristicMap[(y * Nx + x) * 2 + 0];
		new_y = field_MMCCharacteristicMap[(y * Nx + x) * 2 + 1];
	}

	v = bilinear_interpolate(new_x, new_y, FIELD_TYPE);
};



//***************************************//
//			helper methods				 //
//***************************************//


// clamp input value due to min/max
void CFD::clamp(int &val, int min, int max) {
	val = val < min ? min : val;
	val = val > max ? max : val;
};

// clamp input value due to min/max
void CFD::clamp(float &val, float min, float max) {
	val = val < min ? min : val;
	val = val > max ? max : val;
};

// Initialize field with input value
void CFD::init_field(float *field, int x, int y, float value) {

	#pragma omp parallel for
	for (int i = 0; i < x*y; i++) field[i] = value;
};

// Used in contructor, copies color value into color field.
void CFD::clone_color_field(int x, int y, const float *source_field, float *target_field){

	#pragma omp parallel for
	for (int i = 0; i < x*y * 3; i++) target_field[i] = source_field[i];
};


// Bilinear interpolation helper for advection
// Returns in vector3, which each 1D, 2D, 3D grid refers to
// (x), (x,y), (x,y,z) values.
vec3 CFD::bilinear_interpolate(float x, float y, int FIELD_TYPE) { 

	float xx = x / dx;		// normalize to scale 1.
	float yy = y / dy;

	int x0 = (int)xx;	// upper left corner position
	int y0 = (int)yy;
	int x1 = x0 + 1;	// lower right corner position
	int y1 = y0 + 1;

	if (x1 == x0) x0 = 0;	// special cases
	if (y1 == y0) y0 = 0;	// avoid div. by zero

	if (x0 <= 0 || y0 <= 0 || x1 > Nx - 1 || y1 > Ny - 1) 		// check boundary
		return vec3(0, 0, 0);
	

	// interpolate by type
	switch (FIELD_TYPE) {
	case DENSITY:
		{
		// 0. get values of other 4 points				//	v00-----v10
		float v00 = field_density[x0 + (y0 * Nx)];		//	 |		 |
		float v01 = field_density[x0 + (y1 * Nx)];		//	 |	vxy  |
		float v10 = field_density[x1 + (y0 * Nx)];		//	 |		 |
		float v11 = field_density[x1 + (y1 * Nx)];		//	v01-----v11
									
		float vxu, vxl;																// x upper, x lower
		vxu = v00*(x1 - xx) / (x1 - x0) + v10*(xx - x0) / (x1 - x0);				// 1. interpolate x -dir
		vxl = v01*(x1 - xx) / (x1 - x0) + v11*(xx - x0) / (x1 - x0);

		float vy = vxu * (y1 - yy) / (y1 - y0) + vxl * (yy - y0) / (y1 - y0);		// 2. interpolate y - dir
		
		return vec3(vy, -1, -1);
		}
	case VELOCITY:	
		{
		vec2 v00(field_velocity[(x0 + y0 * Nx) * 2], field_velocity[(x0 + y0 * Nx) * 2 + 1]);
		vec2 v01(field_velocity[(x0 + y1 * Nx) * 2], field_velocity[(x0 + y1 * Nx) * 2 + 1]);
		vec2 v10(field_velocity[(x1 + y0 * Nx) * 2], field_velocity[(x1 + y0 * Nx) * 2 + 1]);
		vec2 v11(field_velocity[(x1 + y1 * Nx) * 2], field_velocity[(x1 + y1 * Nx) * 2 + 1]);

		vec2 vxu, vxl;
		vxu.x = v00.x*(x1 - xx) / (x1 - x0) + v10.x*(xx - x0) / (x1 - x0);
		vxu.y = v00.y*(x1 - xx) / (x1 - x0) + v10.y*(xx - x0) / (x1 - x0);
		vxl.x = v01.x*(x1 - xx) / (x1 - x0) + v11.x*(xx - x0) / (x1 - x0);
		vxl.y = v01.y*(x1 - xx) / (x1 - x0) + v11.y*(xx - x0) / (x1 - x0);

		vec3 vy;
		vy.x = vxu.x * (y1 - yy) / (y1 - y0) + vxl.x * (yy - y0) / (y1 - y0);
		vy.y = vxu.y * (y1 - yy) / (y1 - y0) + vxl.y * (yy - y0) / (y1 - y0);
		vy.z = -1;

		return vy;
		}
	case COLOR:
		{												
		color c00(field_color[(x0 + y0 * Nx) * 3 + 0], field_color[(x0 + y0 * Nx) * 3 + 1], field_color[(x0 + y0 * Nx) * 3 + 2]);
		color c01(field_color[(x0 + y1 * Nx) * 3 + 0], field_color[(x0 + y1 * Nx) * 3 + 1], field_color[(x0 + y1 * Nx) * 3 + 2]);
		color c10(field_color[(x1 + y0 * Nx) * 3 + 0], field_color[(x1 + y0 * Nx) * 3 + 1], field_color[(x1 + y0 * Nx) * 3 + 2]);
		color c11(field_color[(x1 + y1 * Nx) * 3 + 0], field_color[(x1 + y1 * Nx) * 3 + 1], field_color[(x1 + y1 * Nx) * 3 + 2]);

		color cxu, cxl;
		cxu.r = c00.r*(x1 - xx) / (x1 - x0) + c10.r*(xx - x0) / (x1 - x0);
		cxu.g = c00.g*(x1 - xx) / (x1 - x0) + c10.g*(xx - x0) / (x1 - x0);
		cxu.b = c00.b*(x1 - xx) / (x1 - x0) + c10.b*(xx - x0) / (x1 - x0);
		cxl.r = c01.r*(x1 - xx) / (x1 - x0) + c11.r*(xx - x0) / (x1 - x0);
		cxl.g = c01.g*(x1 - xx) / (x1 - x0) + c11.g*(xx - x0) / (x1 - x0);
		cxl.b = c01.b*(x1 - xx) / (x1 - x0) + c11.b*(xx - x0) / (x1 - x0);

		color cy;
		cy.r = cxu.r * (y1 - yy) / (y1 - y0) + cxl.r * (yy - y0) / (y1 - y0);
		cy.g = cxu.g * (y1 - yy) / (y1 - y0) + cxl.g * (yy - y0) / (y1 - y0);
		cy.b = cxu.b * (y1 - yy) / (y1 - y0) + cxl.b * (yy - y0) / (y1 - y0);

		return cy;
		}
	}
};

