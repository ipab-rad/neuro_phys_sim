#include <Box2D/Box2D.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "numpy.hpp"

// Rand
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstdlib>
#include <chrono>
#include <thread>
#include <fstream>
#include <sys/stat.h>   /* stat */

using namespace cv;
using namespace std;
using namespace aoba;

typedef struct {
	Mat c_iter_mat;
	Mat last_iter_mat;
	Mat c_iter_mat_bin_world;
	Mat last_iter_mat_bin_world;

	Mat c_binary_obj;
	Mat last_binary_obj;

	float mass;
	float restitution;
	float friction;
	b2Vec2 c_v;
	b2Vec2 last_v;
	// last angle_v, a; body->GetAngularVelocity()
	b2Vec2 c_a;
	b2Vec2 last_a;

	b2Vec2 c_position;
	b2Vec2 last_position;
	float32 c_angle;
	float32 last_angle;
} BodyData;

const int ENUM_DATALAYERS_COUNT = 8; // missing the angular vel and acc
enum DataLayers {world, obj, mass, restitution, v_x, v_y, a_x, a_y};
bool COLLISIONS_OFF = false;

const int max_sides = 10;
const int max_sides_point_spread = 5;
const float scale = 4.0;
const b2Vec2 alpha = b2Vec2(1, -1);

const Size IMAGE_SIZE(480, 480);

float simDist2Draw(float dist) {
	return dist * scale;
}

inline Point2f b2Vec22Point2f(const b2Vec2 v) {
	return Point2f(v.x, v.y);
}

inline Vec2f b2Vec22Vec2f(const b2Vec2 v) {
	return Vec2f(v.x, v.y);
}

inline bool exists_test(const std::string& name) {
	struct stat buffer;
	return (stat (name.c_str(), &buffer) == 0);
}


b2Vec2 simPosition2Draw(const b2Vec2 v, const Size& size = IMAGE_SIZE) {
	b2Vec2 offset(size.width / 2.0, size.height * 0.9);
	return b2Vec2(v.x * alpha.x, v.y * alpha.y) * scale + offset;
}

Mat getBodyImage(const b2Vec2 pos, const Mat& frame) {
	int side = 8;
	int s = simDist2Draw(side);
	Point p = b2Vec22Point2f(simPosition2Draw(pos));
	Rect roi(p.x - s * 0.5, p.y - s * 0.5, s, s);
	return frame(roi);
}

void drawCircle(b2Vec2 pos, float r, Mat& frame) {
	Point2f offset(frame.size().width / 2.0, frame.size().height * 0.9);
	// cout << "Pos: " << pos.x << pos.y << endl;
	circle(frame, b2Vec22Point2f(simPosition2Draw(pos, frame.size())),
	       simDist2Draw(r), CV_RGB(128, 128, 128), CV_FILLED);

}

void drawVec2Poly(vector<b2Vec2> vs, Mat& frame,
                  double scale,
                  Point2f alpha = Point2f(1, -1)) {
	// line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
	// line(frame, Point(0, 0), Point(v.x, v.y), CV_RGB(0, 0, 10), 1);
	// cout << "Position - x: " << v.x << " y: " << v.y << endl;
	Point2f offset(frame.size().width / 2.0, frame.size().height * 0.9);

	// TODO: Conver to draw polygon with CV_FILLED
	std::vector<Point> conv_points;
	for (int i = 0; i < vs.size(); ++i) {
		conv_points.push_back(b2Vec22Point2f(simPosition2Draw(vs[i], frame.size())));
	}
	Scalar colour = CV_RGB(128, 128,
	                       128); // abs(float(rand()) / RAND_MAX - 0.1) * 0 +
	fillConvexPoly(frame, &conv_points[0], conv_points.size(), colour);


	for (int i = 0; i < vs.size(); ++i) {
		line(frame, (Point2f(vs[i].x * alpha.x, vs[i].y * alpha.y) * scale + offset),
		     (Point2f(vs[(i + 1) % vs.size()].x * alpha.x,
		              vs[(i + 1) % vs.size()].y * alpha.y)* scale + offset),
		     CV_RGB(80, 80, 80), 1);
	}
}

void drawPolyShape(b2PolygonShape* poly, b2Body* bd, Mat& frame) {
	int count = poly->GetVertexCount();
	std::vector<b2Vec2> vects;

	for (int i = 0; i < count; i++) {
		b2Vec2 vert = bd->GetWorldPoint(poly->GetVertex(i));
		vects.push_back(vert);
		// drawVec2Poly(vert * 10 + 160, frame);
	}
	drawVec2Poly(vects, frame, scale);
	// cout << "-----" << endl;

	//verts now contains world co-ords of all the verts
}

// If OpenCV version <= 2.4.9, else remove the defintion
void arrowedLine(Mat& img, Point pt1, Point pt2,
                 const Scalar& color,
                 int thickness = 1, int line_type = 8,
                 int shift = 0, double tipLength = 0.1) {
	const double tipSize = norm(pt1 - pt2) *
	                       tipLength; // Factor to normalize the size of the tip depending on the length of the arrow

	line(img, pt1, pt2, color, thickness, line_type, shift);

	const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );

	Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
	        cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
	line(img, p, pt2, color, thickness, line_type, shift);

	p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
	p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
	line(img, p, pt2, color, thickness, line_type, shift);
}

void drawForceArrow(Mat& frame, b2Vec2 force_dir) {
	int len = 100;
	Point st(IMAGE_SIZE.width / 2, IMAGE_SIZE.height * 0.1);
	Point dir(st.x + force_dir.x * len,
	          st.y - force_dir.y * len); // Gravity is along y
	arrowedLine(frame, st, dir, CV_RGB(0, 255, 0), 3); // line
}

inline void cropSmallBinaryFloatImage(b2Body* bd,
                                      const Mat& frame,
                                      Mat& binaryf_crop) {
	Mat small_frame = getBodyImage(bd->GetWorldCenter(), frame);
	Mat small_crop;
	cv::cvtColor(small_frame, small_crop, CV_BGR2GRAY);
	small_crop = (small_crop != 255);
	small_crop.convertTo(binaryf_crop, CV_32FC1, 1.0 / 255.0);
}

fstream fileStream;
void saveToFile(BodyData customdata) {

}


void createRandomBody(b2World& world, bool isBox = true) {
	// Define the dynamic body. We set its position and call the body factory.
	b2BodyDef bodyDef;
	bodyDef.type = b2_dynamicBody;
	float x = rand() % 40 - 20;
	float y = rand() % 5 + 20;
	bodyDef.position.Set(x, y);
	b2Body* body = world.CreateBody(&bodyDef);

	BodyData* customdata = new BodyData;

	// Define another box shape for our dynamic body.
	b2Shape* dynamicShape;
	if (isBox) {
		dynamicShape = new b2PolygonShape;
		((b2PolygonShape*)(dynamicShape))->SetAsBox(1.0f, 1.0f);
		std::cout << "Setting as box!" << std::endl;
	} else {
		int count = int(float(rand()) / RAND_MAX * max_sides); // Max 10 sides
		if (count >= 3) {
			// Make a polygon
			std::vector<Point> random_points, hull;
			do {
				int last_hull_size = 0;
				int attempts_left = max_sides *
				                    max_sides; // max_attempts to generate good points
				do {
					if (random_points.size() > count &&
					        last_hull_size - count < hull.size() - count)
						// if new hull was not better than the old, remove hte point
						random_points.pop_back();
					--attempts_left;
					float xpos = float(rand()) / RAND_MAX * max_sides_point_spread -
					             max_sides_point_spread / 2;
					float ypos = float(rand()) / RAND_MAX * max_sides_point_spread -
					             max_sides_point_spread / 2;
					random_points.push_back(Point(xpos, ypos));
					last_hull_size = hull.size(); // hull size without new point
					convexHull(random_points, hull, true);
					// cout << "size: " << random_points.size() << " " << hull.size() << " " << count << endl;
					// random_points = hull;
					// cout << "Getting points! " << random_points.size() << " " << hull.size()
					//      << " " << attempts_left  << " for " << count << std::endl;
				} while (count >= hull.size() && attempts_left > 0);

			} while (contourArea(hull) < max_sides * max_sides * 0.05f);
			b2Vec2* points = new b2Vec2[hull.size()];
			for (int i = 0; i < hull.size(); ++i) {
				points[i] = b2Vec2(hull[i].x, hull[i].y);
			}
			dynamicShape = new b2PolygonShape;
			((b2PolygonShape*)(dynamicShape))->Set(points, hull.size());
		} else {
			// Make a circle
			dynamicShape = new b2CircleShape;
			((b2CircleShape*)(dynamicShape))->m_p.Set(0, 0); //position,
			//relative to body positionSet(points, hull.size());
			((b2CircleShape*)(dynamicShape))->m_radius =
			    std::max(float(rand()) / RAND_MAX * max_sides_point_spread / 2.0f, 1.0f);
		}
	}
	// std::cout << "Created the shape." << std::endl;

	// Define the dynamic body fixture.
	b2FixtureDef fixtureDef;
	fixtureDef.shape = dynamicShape;

	// Set the box density to be non-zero, so it will be dynamic.
	fixtureDef.density = (float(rand()) / RAND_MAX) * 25 + 5; //10.0f;
	fixtureDef.restitution = float(rand()) / RAND_MAX * 0.5;

	// Override the default friction.
	fixtureDef.friction = 0.3f;

	if (COLLISIONS_OFF) {
		// Specify collision masks
		fixtureDef.filter.categoryBits = 0x0002;
		fixtureDef.filter.maskBits = 0x0002;
	}

	// Add the shape to the body.
	body->CreateFixture(&fixtureDef);

	// Add custom data to the body
	customdata->mass = body->GetMass();
	customdata->restitution = fixtureDef.restitution;
	customdata->friction = fixtureDef.friction;
	customdata->last_v = b2Vec2(0, 0);
	customdata->last_a = b2Vec2(0, 0);
	customdata->last_position = b2Vec2(0, 0);
	customdata->last_angle = 0;
	body->SetUserData(customdata); // GetUserData
}

void createGroundBox(b2World& world,
                     float x_pos, float y_pos,
                     float width, float height,
                     float angle = 0) {
	// Define the ground body.
	b2BodyDef groundBodyDef;
	groundBodyDef.position.Set(x_pos, y_pos);
	if (COLLISIONS_OFF) {
		groundBodyDef.active = false;
	}

	b2Body* groundBody = world.CreateBody(&groundBodyDef);

	// Define the ground box shape.
	b2PolygonShape groundBox;

	// The extents are the half-widths of the box.
	groundBox.SetAsBox(width / 2.0f, height / 2.0f, b2Vec2(0, 0), angle);

	// Add the ground fixture to the ground body.
	groundBody->CreateFixture(&groundBox, 0.0f);
}


int main(int argc, char** argv) {
	B2_NOT_USED(argc);
	B2_NOT_USED(argv);
	// srand(time(NULL));
	if (argc < 2) {
		cout << "You need to specify a seed in CLI. e.g. `./realPhys 3`"  << endl;
	}
	for (int seed_id = 1; seed_id < argc; ++seed_id) {
		int seed = atoi(argv[seed_id]);
		cout << "Using seed: " << seed << endl;
		srand(seed);

		// Define the gravity vector.
		// b2Vec2 gravity(0.0f, -9.8f);
		b2Vec2 gravity(0.0f, -1.7f);

		// Create the world
		b2World world(gravity);

		// ground box
		createGroundBox(world, 0.0f, 0.0f, 64, 1);
		createGroundBox(world, -32.0f, 32.0f, 1, 64);
		createGroundBox(world, 32.0f, 32.0f, 1, 64);
		createGroundBox(world, 0.0f, 64.0f, 64, 1);

		// col obj
		createGroundBox(world, 10.0f, 10.0f, 0.5, 10, -3.14 / 3.0);
		createGroundBox(world, -15.0f, 11.0f, 0.5, 10, 3.14 / 2.80);
		createGroundBox(world, -1.0f, 16.5f, 0.5, 6, -3.14 / 2.35);
		createGroundBox(world, 3.0f, 15.0f, 0.5, 6, 3.14 / 3.9);

		createRandomBody(world);

		// Use 60 Hz
		float32 timeStep = 1.0f / 60.0f;
		int32 velocityIterations = 6;
		int32 positionIterations = 2;
		b2Vec2 force_dir; float max_force = 2000.0f;
		force_dir.Set(0, 0);
		unsigned long long data_obj_id = 0;
		// string filename = "sim_data.data"; // .bin
		// fileStream.open(filename, ios::out);

		// Params
		bool save_only_large_acc = false; double acc_thr = 5;
		bool drawPredictions = true;
		bool should_record_video = true;
		bool use_prediction_velocity = true;
		// string data_dir = "data/";
		// string data_dir = "/media/daniel/8a78d15a-fb00-4b6e-9132-0464b3a45b8b/simdata3/";
		// string data_dir = "/home/daniel/code/neuro_phys_sim/build/datatest/";
		string data_dir = "datapred/";
		// string data_dir = "datacol/";
		// string data_dir = "/media/daniel/8a78d15a-fb00-4b6e-9132-0464b3a45b8b/datacol/";


		Mat traj = Mat::ones(IMAGE_SIZE, CV_8UC3);
		traj = CV_RGB(255, 255, 255);

		// Video
		VideoWriter outputVideo;	// Open the output
		if (should_record_video) {
			outputVideo.open("video.avi", CV_FOURCC('M', 'J', 'P', 'G') , 100 ,
			                 Size(480, 480), true);
			if (!outputVideo.isOpened()) {
				cout  << "Could not open the output video for write " << endl;
				return -1;
			}
		}

		// Main loop
		for (int32 i = 0; i < 10 * 60 * 7; ++i) {
			// Run the physics
			double t = (double)getTickCount();
			world.Step(timeStep, velocityIterations, positionIterations);

			// Extract data
			Mat frame = Mat::ones(IMAGE_SIZE, CV_8UC3);
			frame = CV_RGB(255, 255, 255);
			Mat world_frame = Mat::zeros(IMAGE_SIZE, CV_32FC1);
			Mat mass_frame = Mat::zeros(IMAGE_SIZE, CV_32FC1);
			Mat restitution_frame = Mat::zeros(IMAGE_SIZE, CV_32FC1);
			Mat v_x_frame = Mat::zeros(IMAGE_SIZE, CV_32FC1);
			Mat v_y_frame = Mat::zeros(IMAGE_SIZE, CV_32FC1);
			Mat a_x_frame = Mat::zeros(IMAGE_SIZE, CV_32FC1);
			Mat a_y_frame = Mat::zeros(IMAGE_SIZE, CV_32FC1);


			for (b2Body* bd = world.GetBodyList(); bd; bd = bd->GetNext()) {
				Mat body_only_frame = Mat::ones(IMAGE_SIZE, CV_8UC3);
				body_only_frame = CV_RGB(255, 255, 255);
				for (b2Fixture* fx = bd->GetFixtureList(); fx; fx = fx->GetNext()) {
					b2Shape* shape = fx->GetShape();
					if (shape->GetType() == b2Shape::e_polygon) {
						b2PolygonShape* poly = (b2PolygonShape*)shape;
						drawPolyShape(poly, bd, frame);
						drawPolyShape(poly, bd, body_only_frame);
					} else if (shape->GetType() == b2Shape::e_circle) {
						b2CircleShape* c = (b2CircleShape*)shape;
						drawCircle(bd->GetWorldPoint(c->m_p), c->m_radius, frame);
						drawCircle(bd->GetWorldPoint(c->m_p), c->m_radius, body_only_frame);
					}
				}

				Mat body_imgb;
				Mat body_gray;
				cv::cvtColor(body_only_frame, body_gray, CV_BGR2GRAY);
				body_gray = (body_gray != 255);
				body_gray.convertTo(body_imgb, CV_32FC1, 1.0 / 255.0); // now it's 0 & 1

				world_frame += body_imgb;

				// Update body data (v, a)
				BodyData* customdata = (BodyData*)bd->GetUserData();
				if (customdata != nullptr) {
					b2Vec2 v = bd->GetLinearVelocity();
					customdata->c_a = (customdata->last_v - v) / timeStep;
					if (abs(customdata->c_a.x) > acc_thr || abs(customdata->c_a.y) > acc_thr) {
						cout << "GENERATED A HUGE ACCELERATION!!" << endl;
					}
					customdata->c_v = v;
					customdata->c_position = bd->GetPosition();
					customdata->c_angle = bd->GetAngle();
					// cout << "Updated custom data: " << customdata->last_a.x << " " <<
					//      customdata->last_a.y << endl;

					// Add to global mats
					mass_frame += body_imgb * customdata->mass;
					restitution_frame += body_imgb * customdata->restitution;
					v_x_frame += body_imgb * customdata->last_v.x;
					v_y_frame += body_imgb * customdata->last_v.y;
					a_x_frame += body_imgb * customdata->last_a.x;
					a_y_frame += body_imgb * customdata->last_a.y;
					// Get needed data
					Mat binaryf_crop;
					cropSmallBinaryFloatImage(bd, body_only_frame, binaryf_crop);
					customdata->c_binary_obj = binaryf_crop;
					bd->SetUserData(customdata);
				} else {
					// Part of surrounding world
					float MAX_MASS = 909090;
					mass_frame += body_imgb * MAX_MASS;
					restitution_frame += body_imgb * 0;
					v_x_frame += body_imgb * 0;
					v_y_frame += body_imgb * 0;
					a_x_frame += body_imgb * 0;
					a_y_frame += body_imgb * 0;

				}

				// Apply a force
				// bd->ApplyForce(b2Vec2(50, 0), bd->GetWorldCenter(), true);
				bd->ApplyForceToCenter(force_dir * max_force, true);
				// bd->ApplyLinearImpulse(force_dir * max_force, bd->GetWorldCenter(), true);
				// bd->ApplyTorque(float(i) / 1000, true);
			}
			// imshow("v_x_frame", v_x_frame);
			// cout << mass_frame << endl;

			// Set fps counter on image
			t = ((double)getTickCount() - t) / getTickFrequency();
			// std::cout << "Times passed in seconds: " << t << std::endl;
			int fps_text_size = 15; char* fps = new char[fps_text_size];
			snprintf(fps, fps_text_size, "FPS: %.2f", 1. / t);
			putText(frame, fps, Point(25, 25), FONT_HERSHEY_PLAIN, 1,
			        CV_RGB(128, 127, 127));
			delete fps;

			// Save and show individual images
			int id = world.GetBodyCount();
			for (b2Body* bd = world.GetBodyList(); bd; bd = bd->GetNext(), --id) {
				// Process data only on bodies that we have custom data for
				BodyData* customdata = (BodyData*)bd->GetUserData();
				if (customdata != nullptr) {
					Mat small_bf_crop; // binary float crop
					cropSmallBinaryFloatImage(bd, frame, small_bf_crop);
					// cout << small_bf_crop << endl;
					// TODO:: ADD to the custom data of the body
					customdata->c_iter_mat_bin_world = small_bf_crop;
					bd->SetUserData(customdata);
					// imshow("small_around", small_bf_crop);
					// imshow(to_string(id) + " id", small_bf_crop);

					// Mat k = getBodyImage(bd->GetWorldCenter(), mass_frame);
					// imshow(to_string(id) + " id", k);

					vector<Mat> channels(ENUM_DATALAYERS_COUNT);
					channels[DataLayers::world] = getBodyImage(bd->GetWorldCenter(), world_frame);
					channels[DataLayers::obj] = customdata->c_binary_obj;
					channels[DataLayers::mass] = getBodyImage(bd->GetWorldCenter(), mass_frame);
					channels[DataLayers::restitution] = getBodyImage(bd->GetWorldCenter(),
					                                                 restitution_frame);
					channels[DataLayers::v_x] = getBodyImage(bd->GetWorldCenter(), v_x_frame);
					channels[DataLayers::v_y] = getBodyImage(bd->GetWorldCenter(), v_y_frame);
					channels[DataLayers::a_x] = getBodyImage(bd->GetWorldCenter(), a_x_frame);
					channels[DataLayers::a_y] = getBodyImage(bd->GetWorldCenter(), a_y_frame);

					Mat merged;
					merge(channels, merged);

					customdata->c_iter_mat = merged;

				}
			}

			// Save any information here
			// The last images have led to the current position of object
			for (b2Body* bd = world.GetBodyList(); bd; bd = bd->GetNext()) {
				// Process data only on bodies that we have custom data for
				BodyData* customdata = (BodyData*)bd->GetUserData();
				if (customdata != nullptr) {
					// Move images around
					Mat data = customdata->last_iter_mat;
					if ((!data.empty()) &&
					        ((save_only_large_acc == false) ||
					         (save_only_large_acc == true && (abs(customdata->c_a.x) > acc_thr ||
					                                          abs(customdata->c_a.y) > acc_thr)))) {
						cout << "Saving datafile" << endl;
						// size_t floatcount = data.total() * data.channels();
						// size_t totalwritten = 0;
						std::vector<float> resp;

						// Image data
						float* dataptr = (float*) data.data;
						// fileStream.write((char*)data.data, floatcount * sizeof(float));
						// totalwritten += floatcount * sizeof(float);
						// Delta Position
						b2Vec2 delta_pos = customdata->last_position - customdata->c_position;
						resp.push_back(delta_pos.x);
						resp.push_back(delta_pos.y);
						// fileStream.write((char*)&delta_pos, sizeof(b2Vec2));
						// totalwritten += sizeof(b2Vec2);
						// Delta orientation
						float delta_angle = customdata->last_angle - customdata->c_angle;
						resp.push_back(delta_angle);
						// fileStream.write((char*)&delta_angle, sizeof(float));
						// totalwritten += sizeof(float);
						// acceleration
						resp.push_back(customdata->c_a.x);
						resp.push_back(customdata->c_a.y);
						// fileStream.write((char*)&customdata->c_a, sizeof(b2Vec2));
						// totalwritten += sizeof(b2Vec2);
						// velocity
						resp.push_back(customdata->c_v.x);
						resp.push_back(customdata->c_v.y);
						// fileStream.write((char*)&customdata->c_v, sizeof(b2Vec2));
						// totalwritten += sizeof(b2Vec2);

						// If drawPredictions save to /datatest/ else /data/
						SaveArrayAsNumpy(data_dir + "data_" + to_string(seed) + "_" +
						                 to_string(data_obj_id) + ".npy",
						                 32, 32, 8, dataptr);
						if (!drawPredictions) {
							SaveArrayAsNumpy(data_dir + "resp_"  + to_string(seed) + "_" +
							                 to_string(data_obj_id) + ".npy",
							                 resp.size(), &resp[0]);
						}

						if (drawPredictions) {
							// Check if the network has executed the simulation
							string filename = data_dir + "pred_" + to_string(seed) + "_"
							                  + to_string(data_obj_id) + ".npy";
							cout << "Attempting to open " << filename << endl;

							// Wait until file is created
							std::chrono::milliseconds timespan(50); // or whatever
							std::this_thread::sleep_for(timespan);
							while (!exists_test(filename)) {
								cout << " File still not created " << endl;
								std::this_thread::sleep_for(timespan);
								// usleep(100); // 100 ms
							}
							bool read_ok = false;
							// Read in data
							int rows, cols;
							std::vector<float> pred_nn_data;
							do {
								try {
									LoadArrayFromNumpy(filename, rows, cols, pred_nn_data);
									read_ok = true;
								} catch (const std::exception&) {
									cout << "Some exception from not being able to open the file" << endl;
									cout << filename << endl;
									read_ok = false;
									std::this_thread::sleep_for(timespan);
								}
							} while (!read_ok);
							std::cout << rows << " " << cols << std::endl;
							std::cout << pred_nn_data[0] << " " << pred_nn_data[1]  << std::endl;

							if (use_prediction_velocity) {
								//Update body velocity
								bd->SetLinearVelocity(b2Vec2(pred_nn_data[0], pred_nn_data[1]));
								bd->SetTransform(bd->GetPosition() + b2Vec2(pred_nn_data[0],
								                                            pred_nn_data[1]) * timeStep,
								                 bd->GetAngle());
							}

							// Draw v arrow	int len = 100;
							Point st = b2Vec22Point2f(simPosition2Draw(customdata->c_position));
							int len = 5;
							Point dir1(st.x + pred_nn_data[0] * len,
							           st.y - pred_nn_data[1] * len); // Gravity is along y
							Point dir2(st.x + customdata->c_v.x * len,
							           st.y - customdata->c_v.y * len); // Gravity is along y
							// Draw lines
							Scalar gt_colour = CV_RGB(10, 210, 0);
							Scalar pred_colour = CV_RGB(240, 10, 40);
							Scalar overlap_colour = CV_RGB(20, 10, 245);
							if (!use_prediction_velocity) {
								arrowedLine(frame, st, dir2, gt_colour, 1); // line
							}
							arrowedLine(frame, st, dir1, pred_colour, 1); // line
							if (i % 30 == 0) {
								if (!use_prediction_velocity) {
									arrowedLine(traj, st, dir2, gt_colour, 1); // line
								}
								arrowedLine(traj, st, dir1, pred_colour, 1); // line
							}
							if (!use_prediction_velocity) {
								line(traj, b2Vec22Point2f(simPosition2Draw(customdata->c_position)),
								     b2Vec22Point2f(simPosition2Draw(customdata->last_position)),
								     CV_RGB(10, 10, 10)); // draw trajectory

								// Draw overlapping pixels
								LineIterator it(frame, st, dir2, 8); // GT vector
								for (int i = 0; i < it.count; i++, ++it) {
									Vec3b val = frame.at<Vec3b>(it.pos());
									if (val[0] == pred_colour[0]) { // If as the prediction
										frame.at<Vec3b>(it.pos()) = Vec3b(overlap_colour[0],
										                                  overlap_colour[1],
										                                  overlap_colour[2]);
									}
								}
							}
							// Draw description
							putText(frame, "Prediction", Point(25, 90), FONT_HERSHEY_PLAIN, 2,
							        pred_colour);
							if (!use_prediction_velocity) {
								putText(frame, "Ground Truth", Point(25, 60), FONT_HERSHEY_PLAIN, 2, gt_colour);
								putText(frame, "Overlap", Point(25, 120), FONT_HERSHEY_PLAIN, 2,
								        overlap_colour);
							}
						}


						++data_obj_id;
					}

					// Update current to last data
					// Move images around
					if (!customdata->c_iter_mat.empty()) {
						customdata->last_iter_mat = customdata->c_iter_mat;
						customdata->last_position = customdata->c_position;
						customdata->last_angle = customdata->c_angle;
						customdata->last_a = customdata->c_a;
						customdata->last_v = customdata->c_v;
					}
				}
			}

			if (drawPredictions) {
				imshow("Traj " + to_string(seed), traj);
				imwrite("traj_seed" + to_string(seed) + ".png", traj);
			}
			drawForceArrow(frame, force_dir);
			imshow("Frame (seed " + to_string(seed) + ")", frame);

			if (should_record_video) outputVideo.write(frame);

			// Special keyboard shortcuts
			char k = waitKey(1);
			if (k == 'q' || k == 'c') return 0;
			if (k == ' ') {
				putText(frame, "Paused", Point(25, 60), FONT_HERSHEY_PLAIN, 2,
				        CV_RGB(255, 127, 127));
				imshow("Frame (seed " + to_string(seed) + ")", frame);
				waitKey(0);
			}
#if 0
			// Change the applied force every so often
			if (i % 250 == 0) {
				force_dir.Set((float(rand()) / RAND_MAX - 0.5),
				              (float(rand()) / RAND_MAX - 0.5));
				// force_dir.Set(1, 1);
				// cout << "Force dir: " << force_dir.x << " " << force_dir.y << endl;
			}
#endif
			// Create a new body every so often
			if (i % (60 * 7) == 0 && i < 20000) createRandomBody(world, false);
		}
		destroyAllWindows(); // clean up windows
	}
	// fs.release();
	// fileStream.close();
	// Clean the custom data? Pointers not removed
	return 0;
}
