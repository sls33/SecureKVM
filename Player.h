#ifndef MPC_ML_PLAYER_H
#define MPC_ML_PLAYER_H

#include "Constant.h"
#include "Mat.h"
#include "util/Randomness.h"

class Player
{
public:
    int id;
    ll128 key, lagrange;
    ll128 reshare_key_prev, reshare_key_next;
    RandomnessObj *rand_prev, *rand_next;
    RandomnessObj *rand_extra;
    Player();
    Player(int id, ll128 key, ll128 lagrange);
    static void init();
    static Mat getMetadata();
};

#endif // MPC_ML_PLAYER_H
