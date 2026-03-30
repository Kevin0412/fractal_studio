#include<stdio.h>
#include<stdlib.h>
#include<complex.h>
#include<math.h>
#include"svpng/svpng.inc"
#include<time.h>

int color_r(double n){
    double a;
    a=128.0-128.0*cos(53.0*n*3.141592653589793);
    if(a<256.0){
        return((int)a);
    }
    else{
        return(255);
    }
}

int color_g(double n){
    double a;
    a=128.0-128.0*cos(27.0*n*3.141592653589793);
    if(a<256.0){
        return((int)a);
    }
    else{
        return(255);
    }
}

int color_b(double n){
    double a;
    a=128.0-128.0*cos(139.0*n*3.141592653589793);
    if(a<256.0){
        return((int)a);
    }
    else{
        return(255);
    }
}

int mandelbrot(complex double c,int max){
    // variety 0
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=n*n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int tri(complex double c,int max){
    // variety 1
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=conj(n*n)+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int boat(complex double c,int max){
    // variety 2
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=cabs(creal(n))+cabs(cimag(n))*I;
        n=n*n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int duck(complex double c,int max){
    // variety 3
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=creal(n)+cabs(cimag(n))*I;
        n=n*n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int bell(complex double c,int max){
    // variety 4
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=cabs(creal(n))-cimag(n)*I;
        n=n*n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int fish(complex double c,int max){
    // variety 5
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=n*n;
        n=cabs(creal(n))+cimag(n)*I;
        n=n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int vase(complex double c,int max){
    // variety 6
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=n*n;
        n=cabs(creal(n))-cimag(n)*I;
        n=n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int bird(complex double c,int max){
    // variety 7
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=n*n;
        n=cabs(creal(n))+cabs(cimag(n))*I;
        n=n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int mask(complex double c,int max){
    // variety 8
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=creal(n)+cabs(cimag(n))*I;
        n=n*n;
        n=cabs(creal(n))+cimag(n)*I;
        n=n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

int ship(complex double c,int max){
    // variety 9
    complex double n=0.0+0.0*I;
    int z;
    for(z=0;z<max;z++){
        n=cabs(creal(n))+cimag(n)*I;
        n=n*n;
        n=cabs(creal(n))-cimag(n)*I;
        n=n+c;
        if(cabs(n)>2){
            return(z);
        }
    }
    return(max);
}

void test_rgb(double m,complex double c,char str[256],int s,int h,int variety){
    unsigned char rgb[s*h*3],*p=rgb;
    int x,y;
    FILE *fp=fopen(str,"wb");
    if(variety==0){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)mandelbrot(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==1){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)tri(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==2){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)boat(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==3){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)duck(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==4){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)bell(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==5){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)fish(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==6){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)vase(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==7){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)bird(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==8){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)mask(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else if(variety==9){
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                double n=((double)ship(cpow(2.718281828459045+0.0*I,m-(double)y/(double)s*3.141592653589793*2.0+(double)(2*x-s+1)/(double)s*3.141592653589793*I)+c,5184)+1.0)/5186.0;
                unsigned char r=(unsigned char)color_r(n);
                unsigned char g=(unsigned char)color_g(n);
                unsigned char b=(unsigned char)color_b(n);
                *p++=(unsigned char)r;  /* R */
                *p++=(unsigned char)g;  /* G */
                *p++=(unsigned char)b;  /* B */
            }
        }
    }else{
        for(y=0;y<h;y++){
            for(x=0;x<s;x++){
                *p++=(unsigned char)(rand()%256);  /* R */
                *p++=(unsigned char)(rand()%256);  /* G */
                *p++=(unsigned char)(rand()%256);  /* B */
            }
        }
    }
    svpng(fp,s,h,rgb,0);
    fclose(fp);
}

int main(int argc, char *argv[]){
    clock_t begintime=clock();
    int variety,s,h;
    double n,real, imag;
    char* filename;
    if (argc == 1){
        n = log(4.0);
        real = 0.0;
        imag = 0.0;
        filename = "DefaultArgumentTestImage.png";
        s=360;
        h=160;
        variety = 0;
    }else if (argc == 7){
        n=atof(argv[1]);
        real=atof(argv[2]);
        imag=atof(argv[3]);
        filename=argv[4];
        s=atof(argv[5]);
        h=atof(argv[6]);
        variety=0;
    }else if (argc == 8){
        n=atof(argv[1]);
        real=atof(argv[2]);
        imag=atof(argv[3]);
        filename=argv[4];
        s=atof(argv[5]);
        h=atof(argv[6]);
        variety=atof(argv[7]);
    }else{
        printf("Invalid Argument");
        return 0;
    }
    test_rgb(n,real+imag*I,filename,s,h,variety);
    clock_t endtime=clock();
    printf("\nRunning Time:%fs\n", ((double)(endtime-begintime))/CLOCKS_PER_SEC);
    return(0);
}
