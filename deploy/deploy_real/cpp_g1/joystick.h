#ifndef JOYSTICK_H
#define JOYSTICK_H

typedef union
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} xKeySwitchUnion;

typedef struct
{
    uint8_t head[2];
    xKeySwitchUnion btn;
    float lx;
    float rx;
    float ry;
    float L2;
    float ly;

    uint8_t idle[16];
} xRockerBtnDataStruct;

#endif
