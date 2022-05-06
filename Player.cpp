#include "Player.h"

Player player[M];
Mat metadata(M, M);

Player::Player() {}

Player::Player(int id, ll128 key, ll128 lagrange)
{
    this->id = id;
    this->key = key;
    /// lagrange polynomial coefficients， TODO： modify for reed_solomn reconstruct
    this->lagrange = lagrange;
}

void Player::init()
{
    ll128 key[M], lagrange[M];
    Mat vandermonde(M, M), vandermonde_inv(M, M), des(M, M), pol(M, M), van_inv2(TN, TN), van2(TN, M);
    for (int i = 0; i < M; i++)
    {
        key[i] = Constant::Util::randomlong();
        key[i] = i + 2;
        vandermonde(i, 1) = key[i];
        vandermonde(i, 0) = 1;

        for (int j = 2; j < M; j++)
        {

            vandermonde(i, j) = vandermonde(i, j - 1) * key[i];

            vandermonde(i, j) = Constant::Util::get_residual(vandermonde(i, j));
        }
    }
    DBGprint("init midway\n");
    vandermonde.print();
    for (int i = 0; i < M; i++)
    {
        lagrange[i] = 1;
        for (int j = 0; j < M; j++)
            if (j != i)
            {

                lagrange[i] = lagrange[i] * key[j];

                lagrange[i] = Constant::Util::get_residual(lagrange[i]);

                ll128 tmp = key[j] - key[i];

                tmp = Constant::Util::get_residual(tmp);
                lagrange[i] = lagrange[i] * Constant::Util::inverse(tmp, MOD);

                lagrange[i] = Constant::Util::get_residual(lagrange[i]);
            }
    }

    if (M == 3)
    {
        // Compute reshare key
        // FIXME: currently this only works for 3-party case!!!
        std::vector<ll128> reshare_keys;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < M; j++)
            {
                if (j != i)
                {
                    ll128 tmp_key = 1;
                    for (int k = 0; k < M; k++)
                    {
                        if ((k != i) && (k != j))
                        {
                            cout << i << ", " << j << "," << k << endl;
                            tmp_key = tmp_key * key[k];
                            tmp_key = Constant::Util::get_residual(tmp_key);
                            ll128 divisor = key[k] - key[j];
                            divisor = Constant::Util::get_residual(divisor);
                            tmp_key = tmp_key * Constant::Util::inverse(divisor, MOD);
                            tmp_key = Constant::Util::get_residual(tmp_key);
                        }
                    }
                    reshare_keys.push_back(tmp_key);
                }
            }
        }
        for (ll128 v : reshare_keys)
            cout << v << " ";
        cout << endl;
        // hack
        player[0].reshare_key_next = reshare_keys[4];
        player[0].reshare_key_prev = reshare_keys[2];
        player[1].reshare_key_next = reshare_keys[0];
        player[1].reshare_key_prev = reshare_keys[5];
        player[2].reshare_key_next = reshare_keys[3];
        player[2].reshare_key_prev = reshare_keys[1];

        // Init 128-bit seed
        for (int i = 0; i < M; i++)
        {
            int next = i + 1 >= M ? 0 : i + 1;
            int prev = i - 1 < 0 ? M - 1 : i - 1;
            cout << "rand_seed/key" + to_string(i) + to_string(next) << endl;
            player[i].rand_next = new RandomnessObj("rand_seed/key" + to_string(i) + to_string(next));

            cout << "rand_seed/key" + to_string(prev) + to_string(i) << endl;
            player[i].rand_prev = new RandomnessObj("rand_seed/key" + to_string(prev) + to_string(i));
        }

        // the player0 has an extra permutation mat
        player[0].rand_extra = new RandomnessObj("rand_seed/key12");
    }

    for (int i = 0; i < M; i++)
    {
        player[i].id = i;
        player[i].key = key[i];
        player[i].lagrange = lagrange[i];
    }
    for (int i = 0; i < M; i++)
    {
        DBGprint("%d, key: %lld, %lld\n", player[i].id, (ll)player[i].key, (ll)player[i].lagrange);
    }
    for (int i = 0; i < M; i++)
    {
        ll128 d;
        d = 1;
        for (int j = 0; j < M; j++)
            if (j != i)
            {

                d = d * (key[j] - key[i]);

                d = Constant::Util::get_residual(d);
            }

        d = Constant::Util::inverse(d, MOD);

        for (int j = 0; j < M; j++)
        {

            vandermonde_inv(i, j) = Constant::Util::power(-1, j) * Constant::Util::cal_perm(key, M - j - 1, i);

            vandermonde_inv(i, j) = Constant::Util::get_residual(vandermonde_inv(i, j));

            vandermonde_inv(i, j) = vandermonde_inv(i, j) * d;

            vandermonde_inv(i, j) = Constant::Util::get_residual(vandermonde_inv(i, j));
        }
    }
    {
        ll128 d;
        d = player[1].key - player[0].key;
        d = Constant::Util::inverse(d, MOD);
        van_inv2(0, 0) = player[1].key;
        van_inv2(1, 1) = 1;
        van_inv2(0, 1) = -1;
        van_inv2(1, 0) = 0 - player[0].key;
        for (int i = 0; i < TN; i++)
            for (int j = 0; j < TN; j++)
            {
                van_inv2(i, j) = van_inv2(i, j) * d;
                van_inv2(i, j) = Constant::Util::get_residual(van_inv2(i, j));
            }
    }

    for (int i = 0; i < M; i++)
    {
        van2(0, i) = 1;
        van2(1, i) = player[i].key;
    }

    for (int i = 0; i < TN; i++)
        des(i, i) = 1;
    vandermonde = vandermonde.transpose();
    pol = vandermonde_inv * des * vandermonde;
    metadata = pol;

    DBGprint("\n");

    DBGprint("init complete\n");
}

Mat Player::getMetadata()
{
    return metadata;
}