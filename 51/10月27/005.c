#include<reg51.h>
#include<intrins.h>
#define uchar unsigned char
#define uint unsigned int
uchar code dis_code[]={0xf9,0xa4,0xb0,0x99,0x92,0x82,0xf8,0x80,0x90,0x88,0xc0};
void delay(uint t)
{
    uchar i;
    while(t--)for(i=0;i<20;i++);
}
void main()
{
    uchar i,j=0x80;
    while(1)
    {
        for(i=0;i<8;i++)
        {
            j=_crol_(j,1);
            P0 = dis_code[i];
            P2 = j;
			
            delay(10);
			P0=0xff;
        }
    }
}