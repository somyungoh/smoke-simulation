/****************************************************************
*
*		Somyung(David) Oh
*
*		Project3
*		- Other Advection Schemes
*		- Viscosity
*		- Surface Tension
*
*		Advanced Topics in Physically Based Modeling
*		2018 Spring, Texas A&M University
*		Instructor - Jerry Tessendorf
*
*
****************************************************************/

#define _CRT_SECURE_NO_WARNINGS

#include <Windows.h>
#include <iostream>
#include <string>
#include <sstream>
#include <GL/glut.h>					// GLUT support library.
#define STB_IMAGE_IMPLEMENTATION		// Image input/output library.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "CFD.h"

// define constants
#define WIDTH			512
#define HEIGHT			512
#define CELL_SIZE		0.25
#define MAP_SIZE		WIDTH * HEIGHT
///#define BRUSH_SIZE		5
#define BRUSH_SIZE		10
#define TSTEP_STEP		0.001
#define GRAVITY_STEP	15

using namespace std;

float	*display_map;		// display map(pixel)
float	*obstruction_map;	// obstruction map

							// live variables
float	time_step;
bool	isSimulate;
bool	render_pressure;

// screen capture
bool	capture_screen;
int		frame;
string captured_file_basename;

// painting stuff
bool	autoPaint;
bool	displayObstruction;
int		paint_mode;
enum { PAINT_OBSTRUCTION, PAINT_SOURCE, PAINT_DIVERGENCE, PAINT_COLOR };
float	scaling_factor;
float	obstruction_brush[BRUSH_SIZE][BRUSH_SIZE];
float	source_brush[BRUSH_SIZE][BRUSH_SIZE];

// mouse xy
int		xmouse_prev, ymouse_prev;

// cfd object!!
CFD		cfd;


//  stb_image reader
void readImage(const char* file_name, float* img) {

	int nChannels;		// number of channels
	int width, height;	// image width & height
	unsigned char *temp_map;	// temporary map for image load

								// flip image vertically when loading
	stbi_set_flip_vertically_on_load(true);

	// load image to temp map
	temp_map = stbi_load(file_name, &width, &height, &nChannels, 0);

	// check failure
	if (temp_map)
		cout << "readImage::Image loaded." << endl;
	else
		cout << "readImage::Failed to load image" << endl;

	// convert to image map
#pragma omp parallel for
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 3;
			img[index + 0] = (float)temp_map[index + 0] / 255.f;	// r
			img[index + 1] = (float)temp_map[index + 1] / 255.f;	// g
			img[index + 2] = (float)temp_map[index + 2] / 255.f;	// b
		}
	}


	delete[] temp_map;
}

void writeImage(const char* file_name, float* img) {

	string dir = "Render";		// saving directory
	string format = "jpg";		// save format
	string out_name = dir + "/" + file_name + "." + format;		// full output name

																// create temporary out_map
	unsigned char *out_map = new unsigned char[WIDTH * HEIGHT * 3];

	// copy data
	for (int y = 0; y <HEIGHT; y++) {
#pragma omp parallel for
		for (int x = 0; x < WIDTH; x++) {
			int index = (y * WIDTH + x) * 3;
			int out_index = ((HEIGHT - y - 1) * WIDTH + x) * 3;
			out_map[out_index + 0] = (int)(img[index + 0] * 255 > 255 ? 255 : img[index + 0] * 255);
			out_map[out_index + 1] = (int)(img[index + 1] * 255 > 255 ? 255 : img[index + 1] * 255);
			out_map[out_index + 2] = (int)(img[index + 2] * 255 > 255 ? 255 : img[index + 2] * 255);
		}
	}

	stbi_write_jpg(out_name.c_str(), WIDTH, HEIGHT, 3, out_map, 100);

	delete[] out_map;
}


//--------------------------------------------------------
//
//  Initialization routines
//
//  
// Initialize all of the fields to zero

void Initialize(float *data, int size, float value) {
#pragma omp parallel for
	for (int i = 0; i<size; i++) { data[i] = value; }
}

void InitializeBrushes()
{
	int brush_width = (BRUSH_SIZE - 1) / 2;
#pragma omp parallel for
	for (int j = -brush_width; j <= brush_width; j++)
	{
		int jj = j + brush_width;
		float jfactor = (float(brush_width) - fabs(j)) / float(brush_width);
		for (int i = -brush_width; i <= brush_width; i++)
		{
			int ii = i + brush_width;
			float ifactor = (float(brush_width) - fabs(i)) / float(brush_width);
			float radius = (jfactor*jfactor + ifactor*ifactor) / 2.0;
			source_brush[ii][jj] = pow(radius, 0.5);
			obstruction_brush[ii][jj] = 1.0 - pow(radius, 1.0 / 4.0);
		}
	}
}

//----------------------------------------------------

void ConvertToDisplay()
{
#pragma omp parallel for
	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			// read color by pixel
			// directly from cfd color field
			int index = (WIDTH * y + x) * 3;
			float obstruction = 1;

			color current_color = cfd.get_field_color(x, y);
			if (displayObstruction)
				obstruction = obstruction_map[index / 3];

			if (render_pressure)
				current_color = color(0, cfd.get_pressure_value(x, y), 0);

			float r, g, b;

			r = current_color.r * obstruction;
			g = current_color.g * obstruction;
			b = current_color.b * obstruction;
			display_map[index + 0] = r * scaling_factor;
			display_map[index + 1] = g * scaling_factor;
			display_map[index + 2] = b * scaling_factor;
		}
	}
}


//------------------------------------------
//
//  Painting and display code
//

void resetScaleFactor(float amount)
{
	scaling_factor *= amount;
}


void DabSomePaint(int x, int y)
{
	int brush_width = (BRUSH_SIZE - 1) / 2;
	int xstart = x - brush_width;
	int ystart = y - brush_width;
	if (xstart < 0) { xstart = 0; }
	if (ystart < 0) { ystart = 0; }

	int xend = x + brush_width;
	int yend = y + brush_width;
	if (xend >= WIDTH) { xend = WIDTH - 1; }
	if (yend >= HEIGHT) { yend = HEIGHT - 1; }


	if (paint_mode == PAINT_OBSTRUCTION)
	{
#pragma omp parallel for
		for (int ix = xstart; ix <= xend; ix++)
		{
			for (int iy = ystart; iy <= yend; iy++)
			{
				int index = (ix + WIDTH*(HEIGHT - iy - 1));

				// add to field
				cfd.add_obstruction(index, obstruction_map[index] * obstruction_brush[ix - xstart][iy - ystart]);
				// update obstruction field
				obstruction_map[index] = obstruction_map[index] * obstruction_brush[ix - xstart][iy - ystart];

			}
		}
	}
	else if (paint_mode == PAINT_SOURCE)
	{
#pragma omp parallel for
		for (int ix = xstart; ix <= xend; ix++)
		{
			for (int iy = ystart; iy <= yend; iy++)
			{
				int index = 3 * (ix + WIDTH*(HEIGHT - iy - 1));

				// get and update original color
				color oldC = cfd.get_field_color(index);
				float r = oldC.r + source_brush[ix - xstart][iy - ystart];
				float g = oldC.g + source_brush[ix - xstart][iy - ystart];
				float b = oldC.b + source_brush[ix - xstart][iy - ystart];

				// add to field
				cfd.set_color(index, color(r, g, b));								// color
				cfd.add_source(index / 3, source_brush[ix - xstart][iy - ystart]);	// source
			}
		}
	}
	return;
}



//----------------------------------------------------
//
//  GL and GLUT callbacks
//
//----------------------------------------------------



void cbDisplay(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, display_map);
	glutSwapBuffers();

	if (capture_screen)
	{
		std::stringstream os; os << frame;
		string dispframe = os.str();
		if (frame < 1000) { dispframe = "0" + dispframe; }
		if (frame < 100) { dispframe = "0" + dispframe; }
		if (frame < 10) { dispframe = "0" + dispframe; }
		string fname = captured_file_basename + "." + dispframe;
		writeImage(fname.c_str(), display_map);
		cout << "Frame written to file " << fname << endl;
	}
}

// animate and display new result
void cbIdle()
{
	if (autoPaint) DabSomePaint(WIDTH / 2, HEIGHT - 30);
	if (isSimulate) cfd.sim_one_step(time_step);
	ConvertToDisplay();
	glutPostRedisplay();
	frame++;
}

void cbOnKeyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case '-': case '_':
		resetScaleFactor(0.9);
		break;

	case '+': case '=':
		resetScaleFactor(1.0 / 0.9);
		break;

	case '1':
		cfd.set_advecrtion(CFD::SEMI_LAGRANGIAN);
		cout << "Input::Using Semi-Lagrangian Advection\n";
		break;

	case '2':
		cfd.set_advecrtion(CFD::MODIFIED_MACCORMACK);
		cout << "Input::Using Modified-MacCormack Advection\n";
		break;

	case 'a':
		autoPaint = !autoPaint;
		break;

	case 'r':
		scaling_factor = 1.0;
		break;

	case 'c':
		capture_screen = !capture_screen;
		break;

	case 'e':
		cout << "surface tension: " << cfd.toogle_surfaceT() << endl;
		break;

	case 'k':
		cout << "kinematic viscosity: " << cfd.toogle_kinematicV() << endl;
		break;

	case 'o':
		paint_mode = PAINT_OBSTRUCTION;
		break;

	case 'O':
		displayObstruction = !displayObstruction;
		break;

	case 'p':
		render_pressure = !render_pressure;
		break;

	case 's':
		paint_mode = PAINT_SOURCE;
		break;

	case 32:		// Ascii code for space bar
		isSimulate = !isSimulate;
		if (isSimulate) cout << "Input::Simulation ON" << endl;
		else cout << "Input::Simulation OFF" << endl;
		break;
	case 'g':
	{
		vec2 g = cfd.add_gravity(vec2(0, -GRAVITY_STEP));
		cout << "Input::Gravity " << g.x << "\t" << g.y << endl;
		break;
	}
	case 'G':
	{
		vec2 g = cfd.add_gravity(vec2(0, GRAVITY_STEP));
		cout << "Input::Gravity " << g.x << "\t" << g.y << endl;
		break;
	}
	case 't':
	{
		time_step -= TSTEP_STEP;
		cout << "Input::TimeStep " << time_step << endl;
		break;
	}
	case 'T':
	{
		time_step += TSTEP_STEP;
		cout << "Input::TimeStep " << time_step << endl;
		break;
	}

	case 'Q':
		cout << "Input::Gauss-Seidel interation ";
		cout << cfd.increase_GS_iteration() << endl;
		break;

	case 'q':
		cout << "Input::Gauss-Seidel interation ";
		cout << cfd.decrease_GS_iteration() << endl;
		break;

	case 'W':
		cout << "Input::IOP interation ";
		cout << cfd.increase_IOP_iteration() << endl;
		break;

	case 'w':
		cout << "Input::IOP interation ";
		cout << cfd.decrease_IOP_iteration() << endl;
		break;

	case 'n':
		cout << "Input::Kinematic Viscosity Strength ";
		cout << cfd.decrease_Kviscosity() << endl;
		break;

	case 'N':
		cout << "Input::Kinematic Viscosity Strength ";
		cout << cfd.increase_Kviscosity() << endl;
		break;

	case 'm':
		cout << "Input::Surface Tension Strength ";
		cout << cfd.decrease_Ktension() << endl;
		break;

	case 'M':
		cout << "Input::Surface Tension Strength ";
		cout << cfd.increase_Ktension() << endl;
		break;

	default:
		break;
	}
}

void cbMouseDown(int button, int state, int x, int y)
{
	if (button != GLUT_LEFT_BUTTON) { return; }
	if (state != GLUT_DOWN) { return; }
	xmouse_prev = x;
	ymouse_prev = y;
	DabSomePaint(x, y);
}

void cbMouseMove(int x, int y)
{
	xmouse_prev = x;
	ymouse_prev = y;
	DabSomePaint(x, y);
}

void PrintUsage()
{
	cout << "=====================================================\n";
	cout << "cfd_paint keyboard choices\n";
	cout << "space	 toogles simulation on/ off\n";
	cout << "1		 use Semi-Lagrangian Advection\n";
	cout << "2		 use Modified MacCormack Advection\n";
	cout << "a       turns on auto paint\n";
	cout << "c       toggles screen capture on/off\n";
	cout << "k       toggles kinematic viscosity on/off\n";
	cout << "e       toggles surface tension on/off\n";
	cout << "s       turns on painting source strength\n";
	cout << "o       turns on painting obstructions\n";
	cout << "O		 turns on obstruction display\n";
	cout << "r       resets brightness to default\n";
	cout << "q/Q	 increase/decrease gauss-seidel interation\n";
	cout << "w/W	 increase/decrease IOP iteration\n";
	cout << "n/N	 increase/decrease kinematic viscosity strength\n";
	cout << "m/M	 increase/decrease surface tension strength\n";
	cout << "g/G     increase/decrease gravity\n";
	cout << "t/T     increase/decrease timestep\n";
	cout << "+/-     increase/decrease brightness of display\n";
	cout << "=====================================================\n";
}


void init_components() {

	frame = 1;									// screen capture
	capture_screen = false;
	render_pressure = false;
	captured_file_basename = "cfd";
	isSimulate = true;

	scaling_factor = 1;							// display maps
	displayObstruction = true;
	display_map = new float[3 * MAP_SIZE];
	obstruction_map = new float[MAP_SIZE];;
	Initialize(display_map, 3 * MAP_SIZE, 0.0);
	Initialize(obstruction_map, MAP_SIZE, 1.0);
	float *tempimage = new float[3 * MAP_SIZE];
	Initialize(tempimage, 3 * MAP_SIZE, 0.0);

	string imagename = "me.jpg";				// image load
	readImage(imagename.c_str(), tempimage);

	InitializeBrushes();						// painting
	autoPaint = false;							// auto-paint
	paint_mode = PAINT_SOURCE;

	time_step = 0.009;							// CFD
	cfd = CFD(WIDTH, HEIGHT, CELL_SIZE, tempimage);

	PrintUsage();

	//deallocate tempimage
	delete[] tempimage;
}

//---------------------------------------------------

int main(int argc, char** argv)
{

	init_components();

	// GLUT routines
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);

	// Open a window 
	char title[] = "Project3 - Somyung Oh";
	glutCreateWindow(title);

	glClearColor(1, 1, 1, 1);

	glutDisplayFunc(&cbDisplay);
	glutIdleFunc(&cbIdle);
	glutKeyboardFunc(&cbOnKeyboard);
	glutMouseFunc(&cbMouseDown);
	glutMotionFunc(&cbMouseMove);

	glutMainLoop();
	return 1;

};
