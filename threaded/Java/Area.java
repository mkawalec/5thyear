// Estimates the area of the Mandelbrot set 

public class Area{

    Estimator est;
 
    public static void main(String argv[]){

	Area area = new Area();
	
        area.go();

    }


    void go(){

        int nPoints = 2000;  //number of points in each dimension 
 
        int nIters = 2000;   // maximum number of iterations in test


        Counter numoutside = new Counter(); 	//counter to record the number of points outside the set 

	est = new Estimator(nPoints,nIters,numoutside); //create an estimator object 

	est.estimate(); //do the estimation 

        double result = 2.0*2.5*1.125*(double)(nPoints*nPoints-numoutside.read())/(double)(nPoints*nPoints); // compute area 

        double error = result/(double)nPoints; // ...and error 

        System.out.println("Area = " + result + " +/- " + error);

    }

    // estimates the number of points outside the set 
    class Estimator{

	int nPoints; // number of points in each dimension 

	int nIters; // maximum number of iterations in test 

	Counter numoutside; // counts number of points outside set 

	//constructor 
	Estimator(int nPoints, int nIters, Counter numoutside){

	    this.nPoints = nPoints;
	    this.nIters  = nIters;
            this.numoutside = numoutside;

	}

	// does the actual estimation 
	public void  estimate() {

	    Point point = new Point(0.0,0.0); 

	    for (int i=0; i<nPoints; i++) {
		for (int j=0; j<nPoints; j++) {
		    point.set(-2.0+2.5*(double)(i)/(double)(nPoints)+1.0e-7,
			      1.125*(double)(j)/(double)(nPoints)+1.0e-7);
		    numoutside.add(point.isOutside());
		}
	    }

	}
    

	// point in the complex plane 
	class Point {

	    double real;        //real part 
	    double imag;        //imaginary part 

	    //constructor 
	    Point (double real, double imag){

		this.real = real;
		this.imag = imag;
	    }

	    //copy constructor 
	    Point (Point aPoint){
		
		this(aPoint.real, aPoint.imag);

	    }  

	    // setter method
	    void set(double real, double imag){

		this.real = real;
		this.imag = imag;

	    }

	    // method to test point to see if it is in the set or not 
	    int isOutside(){

		Point z = new Point(this);

		int iter = 0;

		while (z.real*z.real+z.imag*z.imag < 4.0) {
		    double  ztemp=(z.real*z.real)-(z.imag*z.imag)+this.real;
		    z.imag=z.real*z.imag*2+this.imag;
		    z.real=ztemp;
		    iter++; 
		    if (iter == nIters) break; 
		}

		if ( iter == nIters) {
		    return 0;
		}
		else {
		    return 1;
		}

	    }
	    
	}

    }

    // simple integer counter 
    class Counter {

	int value = 0;
 
	// add increment to counter 
	void add(int increment){

	    this.value += increment;
	}

	// read counter 
        int read(){
	    
	    return this.value; 

	}

    }
}