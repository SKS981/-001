#include<reg51.h>
#include<intrins.h>
#define uchar  unsigned char
#define uint unsigned int 

void delay_ms(uint i)
{
	uchar j;
	for(;i>0;i--)
		for(j=0;j<120;j++);
}
uint delay_k[]={800,700,300,200,100,80,50,20};
void main()
{
	uchar t;
	P1=0xFE;
	while(1)
	{	
	P1=P1;
	for(t=0;t<8;t++)
	{
		delay_ms(2*delay_k[t]+10);
		P1=_crol_(P1,1);
		}
	}
}
