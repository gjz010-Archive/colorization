#include <cstdio>
#include <vector>
#include <Dense>
#include <Sparse>
#include <map>
#include <Eigen/SparseExtra>
#include "bitmap_image.hpp"
#define DEBUG
#ifdef DEBUG
#define debug printf
#else
#define debug
#endif



typedef double real;
using std::vector;
struct PixelLoc {
	int index;
	int t;
	real weight;
};

struct Frame {
	vector<real> color_y;
	vector<real> color_u;
	vector<real> color_v;
	vector<real> vx;
	vector<real> vy;
	vector<vector<PixelLoc> >  neighbour;
	Frame(int size) {
		color_y.resize(size);
		neighbour.resize(size);
	}
	int solveNeighbours(int w, int h, int t, Frame* next) {
		int index = 0;
		int n = 0;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				for (int i = -1; i < 2; i++) {
					for (int j = -1; j < 2; j++) {
						if ((x + j >= 0) && (x + j < w) && (y + i >= 0) && (y + i < h) && (i!=0 || j!=0)) {
							int new_index = (y + i)*w + x + j;
							PixelLoc loc = { new_index, t, 0 };
							neighbour[index].push_back(loc);
							n++;
						}
					}
				}
				if (next) {
					int px = round(vx[index] + x);
					int py = round(vy[index] + y);
					for (int i = -1; i < 2; i++) {
						for (int j = -1; j < 2; j++) {
							if ((px + j >= 0) && (px + j < w) && (py + i >= 0) && (py + i < h)) {
								int new_index = (py + i)*w + px + j;
								PixelLoc loc = { new_index, t + 1, 0 };
								neighbour[index].push_back(loc);
								loc = { index, t, 0 };
								next->neighbour[new_index].push_back(loc);
								n += 2;
							}
						}
					}
				}
				index++;
			}
		}
		return n;
	}
	void solveOpticalFlow(int w, int h, const Frame* prev, const Frame* next) {
		vector<real> ixs, iys, its;
		int index = 0;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				real ix, iy, it;
				if (x != 0 && x!=w-1) {
					ix=((color_y[index + 1] - color_y[index - 1]) / 2.0);
				}
				else if (x == 0) {
					ix=(color_y[index + 1] - color_y[index]);
				}
				else {
					ix=(color_y[index] - color_y[index - 1]);
				}
				if (y != 0 && y != h - 1) {
					iy=((color_y[index + h] - color_y[index - h]) / 2.0);
				}
				else if (y == 0) {
					iy=(color_y[index + h] - color_y[index]);
				}
				else {
					iy=(color_y[index] - color_y[index - h]);
				}
				if (prev != NULL && next!=NULL) {
					it=((next->color_y[index] - prev->color_y[index]) / 2.0);
				}
				else if (next != NULL) {
					it=((next->color_y[index] - color_y[index]));
				}
				else if(prev!=NULL){
					it=((color_y[index] - prev->color_y[index]));
				}
				else {
					it=(0);
				}
				ixs.push_back(ix);
				iys.push_back(iy);
				its.push_back(it);
				index++;
			}
		}
		index = 0;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				real sum_ix2 = ixs[index]*ixs[index];
				real sum_ixiy = ixs[index] * iys[index];
				real sum_iy2 = iys[index] * iys[index];
				real sum_ixit = ixs[index] * its[index];
				real sum_iyit = iys[index] * its[index];
				for (int i = -1; i < 2; i++) {
					for (int j = -1; j < 2; j++) {
						if ((x + j >= 0) && (x + j < w) && (y + i >= 0) && (y + i < h)) {
							int new_index = (y + i)*h + x + j;
							sum_ix2 += ixs[new_index] * ixs[new_index];
							sum_iy2 += iys[new_index] * iys[new_index];
							sum_ixiy += ixs[new_index] * iys[new_index];
							sum_ixit += ixs[new_index] * its[new_index];
							sum_iyit += iys[new_index] * its[new_index];
						}
					}
				}
				Eigen::Matrix2d A;
				Eigen::Vector2d b;
				A << sum_ix2, sum_ixiy, sum_ixiy, sum_iy2;
				b << -sum_ixit, -sum_iyit;
				Eigen::Vector2d vec = A.colPivHouseholderQr().solve(b);
				vx.push_back(vec.x());
				vy.push_back(vec.y());
				index++;
			}
		}

	}

};
struct Annotation {
	vector<real> u;
	vector<real> v;
	vector<bool> mask;
	bool valid;
	Annotation(int size) {
		u.resize(size);
		v.resize(size);
		mask.resize(size);
		valid = true;
	}
	Annotation() {
		valid = false;
	}

};
struct Video {
	vector<Frame> frames;
	vector<Annotation> annotations;
	unsigned int n;
	Video(int w, int h, int t) : w(w), h(h), t(t) {
		frames.resize(t, Frame(w*h));
		annotations.resize(t, Annotation());
		n = w*h*t;
	}
	void setAnnotation(int frame, const Annotation& annotation) {
		annotations[frame] = annotation;
	}
	void solveNeighbours() {
		for (int i = 0; i < frames.size(); i++) {
			Frame* prev = i >= 1 ? &frames[i - 1] : nullptr;
			Frame* next = (i < frames.size() - 1) ? &frames[i + 1] : nullptr;
			frames[i].solveOpticalFlow(w, h, prev, next);
			n += frames[i].solveNeighbours(w, h, i, next);
		}
	}
	void solveWeights() {
		for (int i = 0; i < frames.size(); i++) {
			
			Frame& f = frames[i];
			debug("Solving weights for frame %d\n", i);
			int j = 0;

			for (auto& pixel : f.neighbour) {
				real u = f.color_y[j];
				int count = pixel.size()+1;
				for (auto& nb : pixel) {
					//debug("%lf ", frames[nb.t].color_y[nb.index]);
					u += frames[nb.t].color_y[nb.index];
				}
				//debug("\n");
				u = u / count;
				real sigma = (f.color_y[j] - u)*(f.color_y[j] - u);
				for (auto& nb : pixel) {
					sigma += (frames[nb.t].color_y[nb.index] - u)*(frames[nb.t].color_y[nb.index] - u);
				}
				sigma = sigma / count;
				
				for (auto& nb : pixel) {
					if (abs(sigma) < 1e-8) nb.weight = 0;
					else 
						//nb.weight = 1 + (frames[nb.t].color_y[nb.index] - u)*(f.color_y[j] - u) / sigma;
						nb.weight = exp(-(pow(frames[nb.t].color_y[nb.index] - f.color_y[j], 2) / ( sigma)));
					//assert(nb.weight < 10000);
					//if (nb.index==15 && nb.t==0) printf("%d 0 %d %lf\n", i, nb.index, nb.weight);
					//debug("%lf %lf %lf\n", u, sigma, nb.weight);
					//nb.weight = 1 + (frames[nb.t].color_y[nb.index] - u)*(f.color_y[j] - u) / sigma;
				}
				j++;
			}
		}
		for (int i = 0; i < frames.size(); i++) {
			const Frame& f = frames[i];
			for (auto& pixel : f.neighbour) {
				for (auto& nb : pixel) {
					assert(nb.weight < 10000);
				}
			}
		}
	}
	/*
	void solveFrames() {
		debug("Start gradient descent.");
		const int size = w*h*t;
		const int area = w*h;
		

		vector<bool> mask;
		mask.resize(size);
		int index = 0;
		for (int s = 0; s < t; s++) {
			vector<real>* u = &frames[s].color_u, *v = &frames[s].color_v;
			u->resize(area,0); v->resize(area,0);
			if (annotations[s].valid) {
				for (int i = 0; i < area; i++, index++) {
					if (annotations[s].mask[i]) {
						mask[index] = annotations[s].mask[i];
						(*u)[index] = annotations[s].u[i];
						(*v)[index] = annotations[s].v[i];
						//debug("%d %lf %lf\n", index, (*u)[index], (*v)[index]);
					}
					else {
						(*u)[index] = 0.5;
						(*v)[index] = 0.5;
					}
					
					
				}
			}
			else {
				index += area;
			}
		}
		for (int i = 0; i < frames.size(); i++) {
			const Frame& f = frames[i];
			for (auto& pixel : f.neighbour) {
				for (auto& nb : pixel) {
					assert(nb.weight < 10000);
				}
			}
		}

		int max_iter = 10000;
		real alpha = 1;
		vector<real> vgradient_u, vgradient_v;
		vgradient_u.resize(size);
		vgradient_v.resize(size);
		real last_max_error = 100000;
		while (max_iter--) {
			int index = 0;
			real max_error = 0;
			
			for (int s = 0; s < t; s++) {
				if (!mask[index]) {
					vector<real>* u = &frames[s].color_u, *v = &frames[s].color_v;
					for (int i = 0; i < area; i++, index++) {
						real gradient_u = (*u)[index];
						real gradient_v = (*v)[index];
						
						for (const auto& neighbour : frames[s].neighbour[i]) {
							int s_index = area*neighbour.t + neighbour.index;
							//if ((*u)[s_index] > 1e-5) debug("Found!");
							gradient_u -= neighbour.weight*((*u)[s_index]);
							gradient_v -= neighbour.weight*((*v)[s_index]);
							int nb = 0;
							for (auto& nnb : frames[neighbour.t].neighbour[neighbour.index]) {
								if (nnb.t == s && nnb.index == i) {
									break;
								}
								nb++;
							}
							//assert(nb < frames[neighbour.t].neighbour.size());
							real wsr = frames[neighbour.t].neighbour[neighbour.index][nb].weight;
							gradient_u -= wsr*((*u)[s_index] - wsr*((*u)[index]));
							gradient_v -= wsr*((*v)[s_index] - wsr*((*v)[index]));
							//debug("%d %d %d %lf %lf %lf %lf\n", index, neighbour.t, neighbour.index, wsr, neighbour.weight, gradient_u, gradient_v);
							assert(wsr < 10000);

						}
						vgradient_u[index] = gradient_u * 2 * alpha;
						vgradient_v[index] = gradient_v * 2 * alpha;
#define MAX(a,b) ((a)>(b)?(a):(b))
						max_error = MAX(abs(max_error), MAX(abs(gradient_u), abs(gradient_v)));
						
					}
					debug("alpha:%lf, Max Error:%lf %lf %lf\n", alpha, max_error, frames[0].color_u[0], frames[0].color_v[0]);
					if (last_max_error < max_error) {
						alpha = alpha *0.999;
					}
					last_max_error = max_error;
					exit(0);
				}
			}
			index = 0;
			for (int s = 0; s < t; s++) {
				for (int i = 0; i < area; i++,index++) {
					//if (vgradient_u[index] > 0.01) debug("Good!\n");
					frames[s].color_u[i] -= vgradient_u[index];
					frames[s].color_v[i] -= vgradient_v[index];
				}
			}
			
		}
	}*/
	void solveFrames() {
		const int size = w*h*t;
		Eigen::SparseMatrix<real, Eigen::RowMajor> mat(size, size);
		//mat.makeCompressed();
		debug("%d\n", n);
		//mat.reserve(n);
		const int area = w*h;
		int index = 0;
		
		Eigen::VectorXd u(size);
		Eigen::VectorXd v(size);
		u.fill(0);
		v.fill(0);
		int m = 0;
		for (int s = 0; s < t; s++) {
			int index_in_pic = 0;
			Annotation* annotation = nullptr;
			if (annotations[s].valid) {
				annotation = &annotations[s];
			}
			for(int y = 0; y < h; y++) {
				

				for (int x = 0; x < w; x++) {
					mat.insert(index, index) = 1;
					m++;
					bool marked = false;
					
					if (annotation!=nullptr) {
						if (annotation->mask[index_in_pic]) {
							u[index] = annotation->u[index_in_pic];
							v[index] = annotation->v[index_in_pic];
							//debug("Mask! %lf %lf\n", u[index], v[index]);
							marked = true;
						}
					}
					
					if (!marked) {
						real norm = 0.0;
						real sum = 0.0;
						for (const auto& neighbour : frames[s].neighbour[index_in_pic]) {
							norm += neighbour.weight;
							sum += neighbour.weight;
						}
						if (norm < 1e-5) norm = 1;
						//debug("%d %lf %lf\n", index, norm, sum);
						for (const auto& neighbour : frames[s].neighbour[index_in_pic]) {
							int p_index = neighbour.t*area + neighbour.index;
							mat.insert(index, p_index) = -neighbour.weight/norm;
							m++;
							
						}

					}

					index++;
					index_in_pic++;
				}
			}
			
		}


		debug("Start Solving! %d %d %d\n", mat.nonZeros(), n, m);
		
		//mat.makeCompressed();
		//debug("Start QR!\n");
		Eigen::saveMarket(mat, "sparse.mtx");
		Eigen::SparseLU <Eigen::SparseMatrix<real, Eigen::RowMajor> > solver;
		solver.compute(mat);
		debug("Solving LR!\n");
		auto nu = solver.solve(u);
		auto nv = solver.solve(v);
		Eigen::saveMarketVector(nu, "nu.mtx");
		Eigen::saveMarketVector(nv, "nv.mtx");
		debug("%d\n", nu.size());
		index = 0;

		for (int s = 0; s < t; s++) {
			debug("Writing frame %d\n", s);
			frames[s].color_u.resize(area);
			frames[s].color_v.resize(area);
			for (int i = 0; i < area; i++) {
				debug("Writing pixel %d\n", index);
				frames[s].color_u[i]=(nu[index]);
				frames[s].color_v[i]=(nv[index]);
				index++;
			}
		}
	}
	
	void solve() {
		solveNeighbours();
		solveWeights();
		solveFrames();
		debug("Everything Done!");
	}

	void addFrame(Frame f, int t) {
		frames[t]=f;
	}
	int w;
	int h;
	int t;
};


Frame readFrame(std::string filename, int* width, int* height) {
	bitmap_image source(filename.c_str());
	if (width) {
		*width = source.width();
		*height = source.height();
	}
	Frame f(source.width()*source.height());
	int index = 0;
	for (int y = 0; y < source.height(); y++) {
		for (int x = 0; x < source.width(); x++, index++) {
			rgb_t color;
			source.get_pixel(x, y, color);
			real r = ((real)(color.red)) / 255.0;
			real g = ((real)(color.green)) / 255.0;
			real b = ((real)(color.blue)) / 255.0;
			real y = 0.299*r + 0.587*g + 0.144*b;
			if (y > 1) y = 1;
			if (y < 0) y = 0;
			f.color_y[index] = y;
			
		}
	}
	return f;
}
Annotation readAnnotation(std::string markname, std::string maskname) {
	bitmap_image mark(markname.c_str());
	bitmap_image mask(maskname.c_str());
	Annotation an(mark.width()*mark.height());
	int index = 0;
	for (int y = 0; y < mark.height(); y++) {
		for (int x = 0; x < mark.width(); x++, index++) {
			rgb_t color;
			mask.get_pixel(x, y, color);
			if (!(color.blue == 255 && color.green == 255 && color.red == 255)) { //Not white
				//debug("Masking %d %d %d %d\n", index, color.blue, color.green, color.red);
				mark.get_pixel(x, y, color);
				real r = ((real)((int)color.red)) / 255.0;
				real g = ((real)((int)color.green)) / 255.0;
				real b = ((real)((int)color.blue)) / 255.0;
				real u = 0.595716*r - 0.274453*g -0.321263*b;
				if (u > 0.5957) u = 0.5957;
				if (u < -0.5957) u = -0.5957;
				real v = 0.211456*r -0.522591*g + 0.311135*b;
				if (v > 0.5226) v = 0.5226;
				if (v < -0.5226) v = -0.5226;
				an.u[index] = u;
				an.v[index] = v;
				an.mask[index] = true;
				
				
			}
			
		}
	}
	return an;
}
inline rgb_t yuv2rgb(real y, real u, real v) {
	real r = y + 0.956295719758948*u + 0.621024416465261*v;
	real g = y - 0.272122099318510*u - 0.647380596825695*v; 
	real b = y - 1.106989016736491*u + 1.704614998364648 * v;
	r = (r < 0 ? 0 : (r > 1 ? 1 : r));
	g = (g < 0 ? 0 : (g > 1 ? 1 : g));
	b = (b < 0 ? 0 : (b >1 ? 1 : b));
	return { (unsigned char)(round(r * 255)), (unsigned char)(round(g * 255)), (unsigned char)(round(b * 255)) };
}
void writeFrame(const Frame& f, std::string filename, int width, int height) {
	debug("start saving\n");
	bitmap_image image(width, height);
	int index = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++, index++) {
			image.set_pixel(x, y, yuv2rgb(f.color_y[index], f.color_u[index], f.color_v[index]));
			
		}
	}
	debug("saving\n");
	image.save_image(filename.c_str());
}
void generateMask(std::string origname, std::string markname, std::string maskname) {
	bitmap_image orig(origname.c_str());
	bitmap_image mark(markname.c_str());
	int width = mark.width();
	int height = mark.height();
	bitmap_image mask(width, height);
	for (int y = 0; y < mark.height(); y++) {
		for (int x = 0; x < mark.width(); x++) {
			rgb_t c1, c2;
			orig.get_pixel(x, y, c1);
			mark.get_pixel(x, y, c2);
			rgb_t mr = {255,255,255};
			rgb_t mc = { 0,0,0 };
			if (c1 == c2) mask.set_pixel(x, y, mr);
			else mask.set_pixel(x, y, mc);
		}
	}
	mask.save_image(maskname.c_str());
}
int main() {
	int width, height;
	Eigen::initParallel();
	debug("%d threads.\n", Eigen::nbThreads());
	generateMask("samples/baby.bmp", "samples/baby_marked.bmp", "samples/baby_generated_mask.bmp");
	Frame f=readFrame("samples/baby.bmp",&width, &height);
	Annotation a = readAnnotation("samples/baby_marked.bmp", "samples/baby_generated_mask.bmp");
	Video v(width, height, 1);
	v.addFrame(f, 0);
	v.setAnnotation(0, a);
	v.solve();
	writeFrame(v.frames[0], "output.bmp", width, height);
	return 0;
}