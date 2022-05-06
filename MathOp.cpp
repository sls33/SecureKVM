#include "MathOp.h"

MathOp::Add_Mat::Add_Mat() {}

MathOp::Add_Mat::Add_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Add_Mat::forward()
{
    reinit();
    *res->getForward() = (*a->getForward()) + (*b->getForward());
}

void MathOp::Add_Mat::back()
{
    backRound++;
    if (!a->getIsBack())
        *a->getGrad() = (*a->getGrad()) + (*res->getGrad());
    if (!b->getIsBack())
        *b->getGrad() = (*b->getGrad()) + (*res->getGrad());
}

MathOp::Minus_Mat::Minus_Mat() {}

MathOp::Minus_Mat::Minus_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Minus_Mat::forward()
{
    reinit();
    *res->getForward() = (*a->getForward()) - (*b->getForward());
}

void MathOp::Minus_Mat::back()
{
    backRound++;
    if (!a->getIsBack())
        *a->getGrad() = (*a->getGrad()) - (*res->getGrad());
    if (!b->getIsBack())
        *b->getGrad() = (*b->getGrad()) - (*res->getGrad());
}

MathOp::SmoothLevel::SmoothLevel() {}

MathOp::SmoothLevel::SmoothLevel(Mat *res, Mat *a)
{
    this->res = res;
    this->a = a;
    reveal = new Reveal(res, a);
    log_appr = new Log_approximate(res, a, BIT_P_LEN, DECIMAL_PLACES);
    init(2, 0);
}

void MathOp::SmoothLevel::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        *res = res->SmoothLevel();
        break;
    case 3:
        log_appr->forward();
        if (log_appr->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::SmoothLevel::back()
{
}

MathOp::Mul_Mat::Mul_Mat() {}

MathOp::Mul_Mat::Mul_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    temp_a = new Mat(a->getForward()->rows(), a->getForward()->cols());
    temp_b = new Mat(b->getForward()->rows(), b->getForward()->cols());
    div2mP_f = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b_a = new Div2mP(temp_a, temp_a, BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b_b = new Div2mP(temp_b, temp_b, BIT_P_LEN, DECIMAL_PLACES);
    init(2, 3);
}

void MathOp::Mul_Mat::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
    {
        *res->getForward() = (*a->getForward()) * (*b->getForward());
    }
    break;
    case 2:
    {
        div2mP_f->forward();
        if (div2mP_f->forwardHasNext())
        {
            forwardRound--;
        }
    }
    break;
    }
}

void MathOp::Mul_Mat::back()
{
    backRound++;
    switch (backRound)
    {
    case 1:
    {
        if (!a->getIsBack())
        {
            b->getForward()->transorder();
            *temp_a = (*res->getGrad()) * (*b->getForward());
            b->getForward()->transorder();
        }
        if (!b->getIsBack())
        {
            a->getForward()->transorder();
            *temp_b = (*a->getForward()) * (*res->getGrad());
            a->getForward()->transorder();
        }
    }
    break;
    case 2:
        if (!a->getIsBack())
        {
            div2mP_b_a->forward();
        }
        if (!b->getIsBack())
        {
            div2mP_b_b->forward();
        }
        if ((!a->getIsBack() && div2mP_b_a->forwardHasNext()) ||
            (!b->getIsBack() && div2mP_b_b->forwardHasNext()))
        {
            backRound--;
        }
        break;
    case 3:
        *a->getGrad() += *temp_a;
        *b->getGrad() += *temp_b;
        break;
    }
}

MathOp::Hada_Mat::Hada_Mat() {}

MathOp::Hada_Mat::Hada_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    temp_a = new Mat(a->getForward()->rows(), a->getForward()->cols());
    temp_b = new Mat(b->getForward()->rows(), b->getForward()->cols());
    div2mP_f = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b_a = new Div2mP(temp_a, temp_a, BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b_b = new Div2mP(temp_b, temp_b, BIT_P_LEN, DECIMAL_PLACES);
    init(2, 3);
}

void MathOp::Hada_Mat::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
    {
        *res->getForward() = (*a->getForward()).dot(*b->getForward());
    }
    break;
    case 2:
    {
        div2mP_f->forward();
        if (div2mP_f->forwardHasNext())
        {
            forwardRound--;
        }
    }
    break;
    }
}

void MathOp::Hada_Mat::back()
{
    backRound++;

    switch (backRound)
    {
    case 1:
    {
        if (!a->getIsBack())
        {
            *temp_a = (*res->getGrad()).dot(*b->getForward());
        }
        if (!b->getIsBack())
        {
            *temp_b = (*res->getGrad()).dot(*a->getForward());
        }
    }
    break;
    case 2:
        if (!a->getIsBack())
        {
            div2mP_b_a->forward();
        }
        if (!b->getIsBack())
        {
            div2mP_b_b->forward();
        }
        if ((!a->getIsBack() && div2mP_b_a->forwardHasNext()) ||
            (!b->getIsBack() && div2mP_b_b->forwardHasNext()))
        {
            backRound--;
        }
        break;
    case 3:
        *a->getGrad() += *temp_a;
        *b->getGrad() += *temp_b;
    }
}

MathOp::Mul_Const_Trunc::Mul_Const_Trunc() {}

MathOp::Mul_Const_Trunc::Mul_Const_Trunc(Mat *res, Mat *a, double b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    this->revlea1 = new Mat(res->rows(), res->cols());
    *revlea1 = *res;
    div2mP = new Div2mP(res, res, BIT_P_LEN, DECIMAL_PLACES);
    reveal = new Reveal(revlea1, revlea1);
    reveal2 = new Reveal(res, res);
    init(2, 0);
}

void MathOp::Mul_Const_Trunc::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        *res = (*a) * (ll)(b * IE);
        break;
    case 2:
        div2mP->forward();
        if (div2mP->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        cout << "local" << endl;
        revlea1->print();
        break;
    case 5:
        div2mP->forward();
        if (div2mP->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        reveal2->forward();
        if (reveal2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 7:
        cout << "communication" << endl;
        res->print();
        break;
    }
}

void MathOp::Mul_Const_Trunc::back() {}

MathOp::Div_Const_Trunc::Div_Const_Trunc() {}

MathOp::Div_Const_Trunc::Div_Const_Trunc(Mat *res, Mat *a, ll128 b)
{
    this->res = res;
    this->a = a;
    ll128 inverse = Constant::Util::get_residual(IE / b);

    this->b = inverse;
    div2mP = new Div2mP(res, res, BIT_P_LEN, DECIMAL_PLACES);
    init(2, 0);
}

void MathOp::Div_Const_Trunc::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        *res = *a * b;
        break;
    case 2:
        div2mP->forward();
        if (div2mP->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::Div_Const_Trunc::back() {}

MathOp::Div_Const_Trunc_Optimized::Div_Const_Trunc_Optimized() {}

MathOp::Div_Const_Trunc_Optimized::Div_Const_Trunc_Optimized(Mat *res, Mat *a, ll b)
{
    this->res = res;
    this->a = a;

    this->exponent = 1;

    if (IE >= b)
    {
        ll128 inverse = Constant::Util::get_residual(IE / b);
        cout << "Inverse: " << inverse << ", b: " << b << endl;
        this->b = inverse;
        div2mP = new Div2mP(res, res, BIT_P_LEN + DECIMAL_PLACES, DECIMAL_PLACES);
    }
    else
    {
        int exponent = log2(b);
        ll residual = b * IE / pow(2, exponent);
        cout << "exponent: " << exponent << ", residual: " << residual << endl;
        this->b = Constant::Util::get_residual(IE * IE / residual);
        div2mP = new Div2mP(res, res, BIT_P_LEN + DECIMAL_PLACES, exponent + DECIMAL_PLACES);
    }
    init(2, 0);
}

void MathOp::Div_Const_Trunc_Optimized::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        //    *res = (*a * (IE / b));
        // init
        *res = *a * b;
        // res->residual();
        break;
    case 2:
        div2mP->forward();
        if (div2mP->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::Div_Const_Trunc_Optimized::back() {}

MathOp::Div_Seg_Const_Trunc::Div_Seg_Const_Trunc() {}

MathOp::Div_Seg_Const_Trunc::Div_Seg_Const_Trunc(Mat *res, Mat *a, Mat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    div2mP = new Div2mP(res, res, BIT_P_LEN, DECIMAL_PLACES);
    init(2, 0);
}

void MathOp::Div_Seg_Const_Trunc::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        ll inverse;
        for (int i = 0; i < b->size(); ++i)
        {
            inverse = Constant::Util::get_residual(IE / b->getVal(i));
            b->setVal(i, inverse);
        }
        *res = (*a).dot(*b);
        break;
    case 2:
        div2mP->forward();
        if (div2mP->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::Div_Seg_Const_Trunc::back() {}

MathOp::Via::Via() {}

MathOp::Via::Via(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    init();
}

void MathOp::Via::forward()
{
    reinit();
    *res->getForward() = *a->getForward();
}

void MathOp::Via::back()
{
    backRound++;
    if (!a->getIsBack())
        *a->getGrad() = (*a->getGrad()) + (*res->getGrad());
}

MathOp::MeanSquaredLoss::MeanSquaredLoss() {}

MathOp::MeanSquaredLoss::MeanSquaredLoss(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    int tmp_r = a->getForward()->rows();
    int tmp_c = a->getForward()->cols();

    tmp_a = new Mat(tmp_r, tmp_c);
    tmp_b = new Mat(tmp_r, tmp_c);
    tmp_res = new Mat(tmp_r, tmp_c);
    reveal_a = new Reveal(tmp_a, a->getForward());
    reveal_b = new Reveal(tmp_b, b->getForward());
    init(0, 1);
}

void MathOp::MeanSquaredLoss::forward() {}

void MathOp::MeanSquaredLoss::back()
{
    backRound++;
    switch (backRound)
    {
    case 1:
        *b->getGrad() = (*b->getGrad()) + (*a->getForward()) - (*b->getForward());
        break;
    case 2:
        reveal_a->forward();
        reveal_b->forward();
        if (reveal_a->forwardHasNext() || reveal_b->forwardHasNext())
        {
            backRound--;
        }
        break;
    case 3:
        cout << "a, b, loss\n";
        tmp_a->print();
        tmp_b->print();
        break;
    }
}

MathOp::CrossEntropy::CrossEntropy() {}

MathOp::CrossEntropy::CrossEntropy(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;

    reveal = new Reveal(res->getForward(), a->getForward());
    init(0, 0);
}

void MathOp::CrossEntropy::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        *res->getForward() = res->getForward()->cross_entropy();
        break;
    }
}

void MathOp::CrossEntropy::back()
{
    backRound++;
    if (!b->getIsBack())
    {
        *b->getGrad() = (*a->getForward() - *b->getForward()) / (b->getForward()->dot(b->getForward()->oneMinus_IE()));
    }
}

MathOp::Similar::Similar() {}

MathOp::Similar::Similar(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init(1, 0);
}

void MathOp::Similar::forward()
{
    reinit();
    res->getForward()->operator()(0, 0) = a->getForward()->equal(*b->getForward()).count();
}

void MathOp::Similar::back() {}

MathOp::Concat::Concat() {}

MathOp::Concat::Concat(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Concat::forward()
{
    reinit();
    Mat::concat(res->getForward(), a->getForward(), b->getForward());
}

void MathOp::Concat::back()
{
    backRound++;
    Mat::reconcat(res->getGrad(), a->getGrad(), !a->getIsBack(), b->getGrad(), !b->getIsBack());
}

MathOp::Hstack::Hstack() {}

MathOp::Hstack::Hstack(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Hstack::forward()
{
    reinit();
    Mat::hstack(res->getForward(), a->getForward(), b->getForward());
}

void MathOp::Hstack::back()
{
    backRound++;
    Mat::re_hstack(res->getGrad(), a->getGrad(), !a->getIsBack(), b->getGrad(), !b->getIsBack());
}

MathOp::Vstack::Vstack() {}

MathOp::Vstack::Vstack(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Vstack::forward()
{
    reinit();

    Mat::concat(res->getForward(), a->getForward(), b->getForward());
}

void MathOp::Vstack::back()
{
    backRound++;
    Mat::reconcat(res->getBack(), a->getBack(), !a->getIsBack(), b->getBack(), !b->getIsBack());
}

MathOp::Div2mP::Div2mP() {}

MathOp::Div2mP::Div2mP(Mat *res, Mat *a, int k, int m)
{
    this->res = res;
    this->a = a;
    this->k = k;
    this->m = m;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    r_B = new Mat[m];
    for (int i = 0; i < m; i++)
    {
        r_B[i].init(tmp_r, tmp_c);
    }
    r = new Mat(tmp_r, tmp_c);
    b = new Mat(tmp_r, tmp_c);
    pRandM = new PRandM(r_nd, r_st, r_B, k, m);
    pRevealD = new RevealD(b, a, r);

    init(5, 0);
}

void MathOp::Div2mP::forward()
{
    reinit();

    switch (forwardRound)
    {
    case 1:
        
        r_nd->clear();
        r_st->clear();
        for (int i = 0; i < m; i++)
        {
            r_B[i].clear();
        }
        r->clear();
        b->clear();
        break;
    case 2:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            pRandM->forward();
            if (pRandM->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 3:
        *r = *r_nd * (1ll << m) + *r_st;
        *r = *a + (1ll << BIT_P_LEN - 1) + *r;
        break;
    case 4:
        pRevealD->forward();
        if (pRevealD->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
        *b = b->mod(1ll << m);

        *res = (*a - (*b - *r_st)) * Constant::Util::inverse(1ll << m, MOD);
        break;
    }
}

void MathOp::Div2mP::back() {}

void MathOp::Div2mP::reset_for_multi_call()
{
    reset();
    pRandM->reset_for_multi_call();
    pRevealD->reset_for_multi_call();
}

MathOp::Reveal::Reveal() {}

MathOp::Reveal::Reveal(Mat *res, Mat *a)
{
    this->res = res;
    this->a = a;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    b = new Mat(tmp_r, tmp_c);
    init(2, 0);
}

/// TODO: add reed_solomn reconstruct
void MathOp::Reveal::forward()
{
    reinit();
    Mat tmp[M];

    switch (forwardRound)
    {
    case 1:
        *b = *a * player[node_type].lagrange;
        broadcase_rep(b);
        break;
    case 2:

        receive_add(b);
        *res = *b;
        break;
    }
}

void MathOp::Reveal::back() {}

MathOp::ReShare::ReShare() {}

/**
 * Note, this is for 2-out-of-3 reshare. p0 and p1 can perform the resharing, whitout the p2.
 * This shall be in the order 01, 12, 20.
 **/
MathOp::ReShare::ReShare(Mat *res, Mat *a, int p0, int p1)
{
    this->res = res;
    this->a = a;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    b = new Mat(tmp_r, tmp_c);
    share0 = new Mat(tmp_r, tmp_c);
    share1 = new Mat(tmp_r, tmp_c);
    this->p0 = p0;
    this->p1 = p1;

    init(3, 0);
}

void MathOp::ReShare::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        // note, in-place shuffle
        {
            Mat dummy(1, 1);
            if (node_type == p0 || node_type == p1)
            {
                if (node_type == p0)
                {
                    *b = *a * player[p0].reshare_key_next;
                }
                else if (node_type == p1)
                {
                    *b = *a * player[p1].reshare_key_prev;
                }
                shares = IOManager::secret_share_mat_data(*b, b->size(), "");

                // distribute share
                for (int i = 0; i < M; i++)
                {
                    if (i != node_type)
                    {
                        socket_io[node_type][i]->send_message(shares[i]);
                    }
                }
            }
            else
            {
                for (int i = 0; i < M; i++)
                {
                    if (i == p0 || i == p1)
                        MathOp::broadcast_share(&dummy, i); //
                }
            }
            break;
        }
    case 2:
    {
        Mat dummy(1, 1);
        if (node_type == p0)
        {
            *share0 = shares[node_type];
            MathOp::receive_share(share1, p1);
            for (int i = 0; i < M; i++)
            {
                if (i != p0 && i != p1)
                    MathOp::receive_share(&dummy, i);
            }
        }
        else if (node_type == p1)
        {
            *share1 = shares[node_type];
            MathOp::receive_share(share0, p0);
            for (int i = 0; i < M; i++)
            {
                if (i != p0 && i != p1)
                    MathOp::receive_share(&dummy, i);
            }
        }
        else
        {
            // Mat tmp(a->rows(), a->cols());
            MathOp::receive_share(share0, p0);
            MathOp::receive_share(share1, p1);
        }
        break;
    }
    case 3:
        *res = *share0 + *share1;
        break;
    }
}

void MathOp::ReShare::back() {}

MathOp::PRandM::PRandM() {}

MathOp::PRandM::PRandM(Mat *r_nd, Mat *r_st, Mat *b_B, int k, int m)
{
    PRandM_init(r_nd, r_st, b_B, k, m);
}

MathOp::PRandM::PRandM(Mat *r_nd, Mat *r_st, int k, int m)
{
    int tmp_r, tmp_c;
    tmp_r = r_nd->rows();
    tmp_c = r_nd->cols();
    b_B = new Mat[m];
    for (int i = 0; i < m; i++)
    {
        b_B[i].init(tmp_r, tmp_c);
    }
    PRandM_init(r_nd, r_st, b_B, k, m);
}

MathOp::PRandM::PRandM(int r, int c, int k, int m)
{
    r_nd = new Mat(r, c);
    r_st = new Mat(r, c);
    b_B = new Mat[m];
    for (int i = 0; i < m; i++)
    {
        b_B[i].init(r, c);
    }
    PRandM_init(r_nd, r_st, b_B, k, m);
}

void MathOp::PRandM::PRandM_init(Mat *r_nd, Mat *r_st, Mat *b_B, int k, int m)
{
    this->r_nd = r_nd;
    this->r_st = r_st;
    this->b_B = b_B;
    this->k = k;
    this->m = m;
    pRandFld = new PRandFld(r_nd, 1ll << (k - m));
    pRandBit = new PRandBit *[m];
    for (int i = 0; i < m; i++)
    {
        pRandBit[i] = new PRandBit(b_B + i);
    }
    init(4, 0);
}

void MathOp::PRandM::forward()
{
    reinit();
    // cout << "forward: " << forwardRound <<endl;
    switch (forwardRound)
    {
    case 1:
        r_st->clear();
        break;
    case 2:
        pRandFld->forward();
        if (pRandFld->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        for (int i = 0; i < m; i++)
        {
            pRandBit[i]->forward();
        }
        for (int i = 0; i < m; i++)
        {
            if (pRandBit[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 4:
        for (int i = 0; i < m; i++)
        {
            *r_st = *r_st + b_B[i] * (1ll << i);
        }
        break;
    }
}

void MathOp::PRandM::back() {}

void MathOp::PRandM::reset_for_multi_call()
{
    reset();
    pRandFld->reset();
    for (int i = 0; i < m; ++i)
    {
        pRandBit[i]->reset_for_multi_call();
    }
}

MathOp::PRandBit::PRandBit() {}

MathOp::PRandBit::PRandBit(Mat *res)
{
    this->res = res;
    int tmp_r, tmp_c;
    tmp_r = res->rows();
    tmp_c = res->cols();
    a = new Mat(tmp_r, tmp_c);
    a_r = new Mat(1, tmp_r * tmp_c + REDUNDANCY);
    a2 = new Mat(tmp_r, tmp_c);
    a2_r = new Mat(1, tmp_r * tmp_c + REDUNDANCY);
    pRandFld = new PRandFld(a_r, MOD);
    mulPub = new MulPub(a2_r, a_r, a_r);
    init(3, 0);
}

void MathOp::PRandBit::forward()
{
    reinit();

    switch (forwardRound)
    {
    case 1:
        pRandFld->forward();
        if (pRandFld->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        mulPub->forward();
        if (mulPub->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        if (a2_r->count() > REDUNDANCY)
        {
            forwardRound = 0;
            break;
        }
        Mat::fill(a2, a2_r, a, a_r);
        *a2 = a2->sqrt_inv();
        *a2 = a2->dot(*a) + 1;
        *res = a2->divideBy2();
        break;
    }
}

void MathOp::PRandBit::back() {}

void MathOp::PRandBit::reset_for_multi_call()
{
    reset();
    pRandFld->reset();
    mulPub->reset();
}

MathOp::MulPub::MulPub() {}

MathOp::MulPub::MulPub(Mat *res, Mat *a, Mat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init(2, 0);
}

void MathOp::MulPub::forward()
{
    reinit();
    Mat tmp[M];

    switch (forwardRound)
    {
    case 1:
        *res = a->dot(*b);
        *res = *res * player[node_type].lagrange;
        for (int i = 0; i < M; i++)
        {
            if (i != node_type)
            {
                tmp[i] = *res;
            }
        }
        broadcast(tmp);
        break;
    case 2:
        receive(tmp);
        for (int i = 0; i < M; i++)
        {
            if (i != node_type)
            {
                *res = *res + tmp[i];
            }
        }
    }
}

void MathOp::MulPub::back() {}

MathOp::PRandFld::PRandFld() {}

MathOp::PRandFld::PRandFld(Mat *res, ll range)
{
    this->res = res;
    this->range = range;
    init(2, 0);
}

void MathOp::PRandFld::forward()
{
    reinit();
    Mat a[M];

    switch (forwardRound)
    {
    case 1:
        int r, c;
        r = res->rows();
        c = res->cols();

        for (int i = 0; i < M; i++)
        {
            a[i].init(r, c);
        }
        random(a, range);
        *res = a[node_type];
        broadcast(a);
        break;
    case 2:
        receive(a);
        for (int i = 0; i < M; i++)
        {
            if (i != node_type)
            {
                *res = *res + a[i];
            }
        }
        break;
    }
}

void MathOp::PRandFld::back() {}

MathOp::Mod2::Mod2() {}

MathOp::Mod2::Mod2(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    r_B = new Mat[1];
    r_B[0].init(tmp_r, tmp_c);
    c = new Mat(tmp_r, tmp_c);
    pRandM = new PRandM(r_nd, r_st, r_B, k, 1);
    reveal = new Reveal(c, c);
    init(5, 0);
}

void MathOp::Mod2::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        r_nd->clear();
        r_st->clear();
        r_B[0].clear();
        c->clear();
        // todo: offline
        break;
    case 2:
        if (OFFLINE_PHASE_ON)
        {
            pRandM->forward();
            if (pRandM->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 3:
        *c = *a + (1ll << BIT_P_LEN - 1) + (*r_nd * 2) + *r_st;
        break;
    case 4:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
        *res = (*c & 1) + *r_st - (*c & 1).dot(*r_st) * 2;
        r_nd->clear();
        r_st->clear();
        r_B[0].clear();
        c->clear();
        break;
    }
}

void MathOp::Mod2::back() {}

MathOp::Mod2D::Mod2D() {}

MathOp::Mod2D::Mod2D(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    r_B = new Mat[1];
    r_B[0].init(tmp_r, tmp_c);
    c = new Mat(tmp_r, tmp_c);
    pRandM = new PRandM(r_nd, r_st, r_B, k, 1);
    reveal = new Reveal(c, c);
    degred = new DegRed(res, res);
    init(4, 0);
}

void MathOp::Mod2D::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        r_nd->clear();
        r_st->clear();
        r_B[0].clear();
        c->clear();

        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            pRandM->forward();
            if (pRandM->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 2:
        *c = *a + (1ll << BIT_P_LEN - 1) + (*r_nd * 2) + *r_st;
        break;
    case 3:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        *res = (*c & 1) + *r_st - (*c & 1).dot(*r_st) * 2;
        r_nd->clear();
        r_st->clear();
        r_B[0].clear();
        c->clear();
        break;
    }
}

void MathOp::Mod2D::back() {}

MathOp::DegRed::DegRed() {}

MathOp::DegRed::DegRed(Mat *res, Mat *a)
{
    this->res = res;
    this->a = a;
    tmp = new Mat[M];
    init(2, 0);
}

void MathOp::DegRed::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        for (int i = 0; i < M; i++)
        {
            if (i == node_type)
            {
                continue;
            }
            tmp[i] = *a * metadata(node_type, i);
        }
        broadcast(tmp);
        break;
    case 2:
        *res = *a * metadata(node_type, node_type);
        receive(tmp);
        for (int i = 0; i < M; i++)
        {
            if (i != node_type)
            {
                *res = *res + tmp[i];
            }
        }
    }
}

void MathOp::DegRed::back() {}

MathOp::PreMulC::PreMulC() {}

MathOp::PreMulC::PreMulC(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;
    int tmp_r, tmp_c;
    tmp_r = a[0].rows();
    tmp_c = a[0].cols();
    r = new Mat[k];
    s = new Mat[k];
    u = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        r[i].init(tmp_r, tmp_c);
        s[i].init(tmp_r, tmp_c);
        u[i].init(tmp_r, tmp_c);
    }
    pRandFld_r = new PRandFld *[k];
    pRandFld_s = new PRandFld *[k];
    pMulPub_st = new MulPub *[k];
    pDegRed = new DegRed *[k];
    for (int i = 0; i < k; i++)
    {
        pRandFld_r[i] = new PRandFld(r + i, MOD);
        pRandFld_s[i] = new PRandFld(s + i, MOD);
        pMulPub_st[i] = new MulPub(u + i, r + i, s + i);
    }

    m = new Mat[k];
    w = new Mat[k];
    z = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        m[i].init(tmp_r, tmp_c);
        w[i].init(tmp_r, tmp_c);
        z[i].init(tmp_r, tmp_c);
    }
    for (int i = 1; i < k; i++)
    {
        pDegRed[i] = new DegRed(w + i, w + i);
    }
    pMulPub = new MulPub *[k];
    for (int i = 0; i < k; i++)
    {
        pMulPub[i] = new MulPub(m + i, w + i, a + i);
    }
    init(9, 0);
}

void MathOp::PreMulC::forward()
{
    reinit();
    // cout << "PreMulC: " << forwardRound << endl;
    switch (forwardRound)
    {
    case 1:
        break;
    case 2:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            for (int i = 0; i < k; i++)
            {
                pRandFld_r[i]->forward();
                pRandFld_s[i]->forward();
            }
            for (int i = 0; i < k; i++)
            {
                if (pRandFld_r[i]->forwardHasNext() || pRandFld_s[i]->forwardHasNext())
                {
                    forwardRound--;
                    break;
                }
            }
        }
        break;
    case 3:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            for (int i = 0; i < k; i++)
            {
                pMulPub_st[i]->forward();
            }
            for (int i = 0; i < k; i++)
            {
                if (pMulPub_st[i]->forwardHasNext())
                {
                    forwardRound--;
                    break;
                }
            }
        }
        break;
    case 4:
        break;
    case 5:
        w[0] = r[0];
        for (int i = 1; i < k; i++)
        {
            w[i] = r[i].dot(s[i - 1]);
        }
        break;
    case 6:
        for (int i = 1; i < k; i++)
        {
            pDegRed[i]->forward();
        }
        for (int i = 1; i < k; i++)
        {
            if (pDegRed[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 7:
        {
            // optimization: make use of memory localization
            Mat inverse_tmp[k];
            std::vector<std::thread> thrds;
            int thread_num = THREAD_NUM;
            int seg_len = ceil(k * 1.0 / thread_num);
            for (int i = 0; i < thread_num; i++)
            {
                thrds.emplace_back(std::thread([this, i, seg_len, &inverse_tmp]()
                                               {
                    for (int j = i*seg_len; j < (i+1)*seg_len && j < k; j++) {
                        inverse_tmp[j] = u[j].inverse();
                    } }));
            }
            for (auto &t : thrds)
                t.join();

            for (int i = 1; i < k; i++)
            {
                w[i] = w[i].dot(inverse_tmp[i - 1]);
            }
            for (int i = 0; i < k; i++)
            {
                z[i] = s[i].dot(inverse_tmp[i]);
            }
        }
        break;
    case 8:
        for (int i = 0; i < k; i++)
        {
            pMulPub[i]->forward();
        }
        for (int i = 0; i < k; i++)
        {
            if (pMulPub[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 9:
        res[0] = a[0];
        for (int i = 1; i < k; i++)
        {
            m[i] = m[i].dot(m[i - 1]);
        }
        for (int i = 1; i < k; i++)
        {
            res[i] = m[i].dot(z[i]);
        }
        break;
    }
}

void MathOp::PreMulC::back() {}

MathOp::BitLT::BitLT() {}

MathOp::BitLT::BitLT(Mat *res, Mat *a, Mat *b_B, int k)
{
    this->res = res;
    this->a = a;
    this->b_B = b_B;
    this->k = k;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    d_B = new Mat[k];
    p_B = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        d_B[i].init(tmp_r, tmp_c);
        p_B[i].init(tmp_r, tmp_c);
    }
    d_B_inverse = new Mat[k];
    p_B_inverse = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        d_B_inverse[i].init(tmp_r, tmp_c);
        p_B_inverse[i].init(tmp_r, tmp_c);
    }
    s = new Mat(tmp_r, tmp_c);
    preMulC = new PreMulC(p_B, d_B_inverse, k);
    pMod2 = new Mod2(res, s, k);
    init(4, 0);
}

void MathOp::BitLT::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        for (int i = 0; i < k; i++)
        {
            d_B[i] = a->get_bit(i) + b_B[i] - a->get_bit(i).dot(b_B[i]) * 2; // d_B[k-1-i] -> d_B[i]
        }
        for (int i = 0; i < k; i++)
        {
            d_B[i] = d_B[i] + 1;
        }
        // inverse input, considering SufMul
        for (int j = 0; j < k; ++j)
        {
            d_B_inverse[j] = d_B[k - 1 - j];
        }
        break;
    case 2:
        preMulC->forward();
        if (preMulC->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        // inverse output, considering SufMul
        for (int j = 0; j < k; ++j)
        {
            p_B_inverse[j] = p_B[k - 1 - j];
        }
        *s = a->get_bit(k - 1).oneMinus().dot(d_B[k - 1] - 1);
        for (int i = 0; i < k - 1; i++)
        {
            *s = *s + a->get_bit(i).oneMinus().dot(p_B_inverse[i] - p_B_inverse[i + 1]);
        }
        break;
    case 4:
        pMod2->forward();
        if (pMod2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::BitLT::back() {}

MathOp::RevealD::RevealD() {}

MathOp::RevealD::RevealD(Mat *res, Mat *a, Mat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    pReveal = new Reveal(res, b);
    pDegRed = new DegRed(a, a);
    init(1, 0);
}

MathOp::RevealD::RevealD(Mat *res, Mat *a)
{
    this->res = res;
    this->a = a;
    pReveal = new Reveal(res, a);
    init(1, 0);
}

void MathOp::RevealD::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        pDegRed->forward();
        pReveal->forward();
        if (pReveal->forwardHasNext() || pDegRed->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::RevealD::back() {}

void MathOp::RevealD::reset_for_multi_call()
{
    reset();
    pReveal->reset();
    pDegRed->reset();
}

MathOp::Mod2m::Mod2m() {}

MathOp::Mod2m::Mod2m(Mat *res, Mat *a, int k, int m)
{
    this->res = res;
    this->a = a;
    this->k = k;
    this->m = m;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    r_B = new Mat[m];
    for (int i = 0; i < m; i++)
    {
        r_B[i].init(tmp_r, tmp_c);
    }
    b = new Mat(tmp_r, tmp_c);
    u = new Mat(tmp_r, tmp_c);
    pRandM = new PRandM(r_nd, r_st, r_B, k, m);
    pRevealD = new RevealD(b, a, b);
    pBitLT = new BitLT(u, b, r_B, m);
    init(6, 0);
}

void MathOp::Mod2m::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        if (OFFLINE_PHASE_ON)
        {
            pRandM->forward();
            if (pRandM->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 2:
        *b = *a + (1ll << (BIT_P_LEN - 1)) + (*r_nd) * (1ll << m) + (*r_st);
        break;
    case 3:
        pRevealD->forward();
        if (pRevealD->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        *b = b->mod(1ll << m);
        break;
    case 5:
        pBitLT->forward();
        if (pBitLT->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        *res = *b - (*r_st) + (*u) * (1ll << m);
        r_nd->clear();
        r_st->clear();
        for (int i = 0; i < m; i++)
        {
            r_B[i].clear();
        }
        u->clear();
        b->clear();
        break;
    }
}

void MathOp::Mod2m::back() {}

MathOp::Div2m::Div2m() {}

MathOp::Div2m::Div2m(Mat *res, Mat *a, int k, int m)
{
    this->res = res;
    this->a = a;
    this->k = k;
    this->m = m;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    b = new Mat(tmp_r, tmp_c);
    pMod2m = new Mod2m(b, a, k, m);
    init(2, 0);
}

void MathOp::Div2m::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        pMod2m->forward();
        if (pMod2m->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        *res = (*a - *b) * Constant::Util::inverse(1ll << m, MOD); 
        b->clear();
        break;

    }
}

void MathOp::Div2m::back() {}

MathOp::LTZ::LTZ() {}

MathOp::LTZ::LTZ(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;

    pDiv2m = new Div2m(res, a, k, k - 1);

    init(2, 0);
}

void MathOp::LTZ::forward()
{
    reinit();

    switch (forwardRound)
    {
    case 1:
        pDiv2m->forward();
        if (pDiv2m->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        // return IE, 0
        // *res = res->opposite() * IE;
        // return 1, 0
        *res = res->opposite();
        break;
    case 5:
        reveal_tmp->forward();
        if (reveal_tmp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        reveal_a->forward();
        if (reveal_a->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 7:
        cout << "----a \n";
        a->print();
    }
}


void MathOp::LTZ::back() {}

MathOp::KOrCL::KOrCL() {}

MathOp::KOrCL::KOrCL(Mat *res, Mat *d_B, int k)
{
    this->res = res;
    this->d_B = d_B;
    this->k = k;
    int tmp_r, tmp_c;
    tmp_r = res->rows();
    tmp_c = res->cols();
    // PRandM
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    r_B = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        r_B[i].init(tmp_r, tmp_c);
    }
    coefficients = {99999999999999948, 33463158713874187, 86737888256286702, 11567549490043411, 99921519514012901, 79328041657849319,
                    71201314937608293, 98977993829566184, 4710723341940302, 29861796160469053, 89279246751037322, 36968306758133112, 58454784438587612,
                    75328931003967543, 86384837554252909, 6683547233687751, 50782373290388091, 86538866472678812, 6325226823853224, 3985933527286092,
                    77424372073292124, 91130324538764507, 30006118432141175, 85528322862801917, 19323778082869053, 55452091436759921,
                    50968410924755803, 15487837657987303, 92765158379360327, 19125538233297694, 64665344039295436, 21511061014140188,
                    38341389469709633, 91138994162178133, 2596434147240363, 18563992726472778, 20314152126546477, 46353985440399574,
                    40509026434273843, 10902356853490174, 78426231192074644, 7357679543159906, 36738669188341968, 97244173271201797,
                    28498238528940992, 19107855296183373, 48624109143330949, 21615215036840124, 31351471151295706, 11808938035990712,
                    27816325723419482, 58708837276119684, 31861430804545569, 85836083318538838, 75971425250598398, 30422588448118748};
    // Random b_i, b_i', c_i = A * b_(i-1) * b_i^-1
    b = new Mat[k];
    b_st = new Mat[k];
    B_pub = new Mat[k];
    B_mul = new Mat[k];
    C = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        b[i].init(tmp_r, tmp_c);
        b_st[i].init(tmp_r, tmp_c);
        B_pub[i].init(tmp_r, tmp_c);
        B_mul[i].init(tmp_r, tmp_c);
        C[i].init(tmp_r, tmp_c);
    }
    pRandFld_b = new PRandFld *[k];
    pRandFld_b_st = new PRandFld *[k];
    mul_pub = new MulPub *[k];
    for (int i = 0; i < k; i++)
    {
        pRandFld_b[i] = new PRandFld(b + i, MOD);
        pRandFld_b_st[i] = new PRandFld(b_st + i, MOD);
        mul_pub[i] = new MulPub(B_pub + i, b + i, b_st + i);
    }

    pDegRed = new DegRed *[k];
    for (int i = 0; i < k; ++i)
    {
        pDegRed[i] = new DegRed(B_mul + i, B_mul + i);
    }

    mul_pub_nd = new MulPub *[k];
    A = new Mat(tmp_r, tmp_c);
    for (int i = 0; i < k; ++i)
    {
        mul_pub_nd[i] = new MulPub(C + i, A, B_mul + i);
    }

    A_pow = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        A_pow[i].init(tmp_r, tmp_c);
    }
    //    pRandM = new PRandM(r_nd, r_st, r_B, k, k);
    tmp1 = new Mat(tmp_r, tmp_c);
    tmp2 = new Mat[k];
    for (int i = 0; i < k; ++i)
    {
        tmp2[i].init(tmp_r, tmp_c);
    }
    reveal_a_pow = new Reveal *[k];
    for (int i = 0; i < k; ++i)
    {
        reveal_a_pow[i] = new Reveal(tmp2 + i, A_pow + i);
    }
    reveal = new Reveal(tmp1, A);
    init(11, 0);
}

void MathOp::KOrCL::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        // Lagrange interpolation f(1) = 0,
        break;
    case 2:
        A->clear();
        for (int i = 0; i < k; ++i)
        {
            *A = *A + d_B[i];
        }
        *A = *A + 1;
        break;
    case 3:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        break;
    case 5:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            for (int i = 0; i < k; i++)
            {
                pRandFld_b[i]->forward();
                pRandFld_b_st[i]->forward();
            }
            for (int i = 0; i < k; i++)
            {
                if (pRandFld_b[i]->forwardHasNext() || pRandFld_b_st[i]->forwardHasNext())
                {
                    forwardRound--;
                    break;
                }
            }
        }
        break;
    case 6:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            for (int i = 0; i < k; i++)
            {
                mul_pub[i]->forward();
            }
            for (int i = 0; i < k; i++)
            {
                if (mul_pub[i]->forwardHasNext())
                {
                    forwardRound--;
                    break;
                }
            }
        }
        break;
    case 7:
        // calculate A^i
        B_mul[0] = b_st[0].dot(B_pub[0].inverse());
        for (int i = 1; i < k; ++i)
        {
            B_mul[i] = b[i - 1].dot(B_pub[i].inverse()).dot(b_st[i]);
        }
        break;
    case 8:
        for (int i = 1; i < k; i++)
        {
            pDegRed[i]->forward();
            if (pDegRed[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 9:
        for (int i = 0; i < k; i++)
        {
            mul_pub_nd[i]->forward();
        }
        for (int i = 0; i < k; i++)
        {
            if (mul_pub_nd[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 10:
        for (int i = 0; i < k; ++i)
        {
            A_pow[i] = b[i];
            for (int j = 0; j <= i; ++j)
            {
                A_pow[i] = A_pow[i].dot(C[j]);
            }
        }
        break;
    case 11:
        res->clear();
        *res = *res + coefficients[0];
        // todo: multiply lagrange coefficients
        for (int i = 1; i <= k; ++i)
        {
            *res += A_pow[i - 1] * coefficients[i];
        }
        break;
    case 12:
        for (int i = 0; i < k; ++i)
        {
            reveal_a_pow[i]->forward();
        }
        for (int i = 0; i < k; i++)
        {
            if (reveal_a_pow[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 13:
        cout << "A-pow\n";
        //            for (int i = 0; i < k; ++i) {
        //                cout << i << endl;
        //                tmp2[i].print();
        //            }
        break;
    }
}

void MathOp::KOrCL::back() {}

MathOp::EQZ::EQZ() {}

MathOp::EQZ::EQZ(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;

    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    r_B = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        r_B[i].init(tmp_r, tmp_c);
    }
    b = new Mat(tmp_r, tmp_c);
    u = new Mat(tmp_r, tmp_c);

    d_B = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        d_B[i].init(tmp_r, tmp_c);
    }
    tmp1 = new Mat(tmp_r, tmp_c);
    reveal_tmp = new Reveal(tmp1, a);
    pRandM = new PRandM(r_nd, r_st, r_B, k, k);
    pReveal = new Reveal(b, b);
    kOrCl = new KOrCL(res, d_B, k);
    init(6, 0);
}

void MathOp::EQZ::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            pRandM->forward();
            if (pRandM->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 2:
        // todo: c'=r' mod 2^k
        *b = *a + (1ll << (BIT_P_LEN)) + (*r_nd) * (1ll << k) + (*r_st); 
        break;
    case 3:
        pReveal->forward();
        reveal_tmp->forward();
        if (pReveal->forwardHasNext() || reveal_tmp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        // cout << "a\n";
        // tmp1->print();
        for (int i = 0; i < k; i++)
        {
            d_B[i] = b->get_bit(i) + r_B[i] - b->get_bit(i).dot(r_B[i]) * 2; // d_B[k-1-i] -> d_B[i]
        }
        break;
    case 5:
        kOrCl->forward();
        if (kOrCl->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        *res = res->oneMinus();
        r_nd->clear();
        r_st->clear();
        for (int i = 0; i < k; i++)
        {
            r_B[i].clear();
        }
        break;
    }
}

void MathOp::EQZ::back() {}

MathOp::EQZ_2LTZ::EQZ_2LTZ() {}

MathOp::EQZ_2LTZ::EQZ_2LTZ(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();

    u_st = new Mat(tmp_r, tmp_c);
    u_st_res = new Mat(tmp_r, tmp_c);
    u_nd = new Mat(tmp_r, tmp_c);
    u_nd_res = new Mat(tmp_r, tmp_c);
    pLTZ_f_1 = new LTZ(u_st_res, u_st, BIT_P_LEN);
    pLTZ_f_2 = new LTZ(u_nd_res, u_nd, BIT_P_LEN);
    reveal = new Reveal(res, res);
    reveal_1 = new Reveal(u_st_res, u_st_res);
    reveal_2 = new Reveal(u_nd_res, u_nd_res);
    init(4, 0);
}

void MathOp::EQZ_2LTZ::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        *u_st = *a;
        *u_nd = a->opposite();
        break;
    case 2:
        pLTZ_f_1->forward();
        pLTZ_f_2->forward();
        if (pLTZ_f_1->forwardHasNext() || pLTZ_f_2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        *u_st_res = u_st_res->oneMinus();
        *u_nd_res = u_nd_res->oneMinus();
        *res = u_st_res->dot(*u_nd_res);
        break;
    case 4:
        reveal->forward();
        reveal_1->forward();
        reveal_2->forward();
        if (reveal->forwardHasNext() || reveal_1->forwardHasNext() || reveal_2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
        cout << "EQZ_2LTZ\n";
        u_st_res->print();
        u_nd_res->print();
        res->print();
        break;
    }
}

void MathOp::EQZ_2LTZ::back() {}

MathOp::ReLU_Mat::ReLU_Mat() {}

MathOp::ReLU_Mat::ReLU_Mat(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    res->setAux(new Mat(tmp_r, tmp_c));
    pLTZ = new LTZ(res->getAux(), a->getForward(), BIT_P_LEN);
    pDiv2mp_f = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    pDiv2mp_b = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);

    tmp_a = new Mat(tmp_r, tmp_c);
    tmp_res = new Mat(tmp_r, tmp_c);
    reveal_a = new Reveal(tmp_a, a->getForward());
    reveal_res = new Reveal(tmp_res, res->getForward());
    init(6, 2);
}

void MathOp::ReLU_Mat::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        pLTZ->forward();
        if (pLTZ->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        *res->getAux() = res->getAux()->oneMinus_IE();
        break;
    case 3:
        *res->getForward() = a->getForward()->dot(*res->getAux());
        break;
    case 4:
        pDiv2mp_f->forward();
        if (pDiv2mp_f->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
        reveal_a->forward();
        reveal_res->forward();
        if (reveal_a->forwardHasNext() || reveal_res->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        cout << "a---\n";
        tmp_a->print();
        cout << "res---\n";
        tmp_res->print();
        break;
    }
}

void MathOp::ReLU_Mat::back()
{
    backRound++;
    switch (backRound)
    {
    case 1:
        *a->getGrad() = res->getGrad()->dot(*res->getAux()) + res->getAux()->oneMinus_IE() * LEAKEY_RELU_BIAS;
        break;
    case 2:
        pDiv2mp_b->back();
        if (pDiv2mp_b->backHasNext())
        {
            backRound--;
        }
        break;
    }
}

MathOp::Sigmoid_Mat::Sigmoid_Mat() {}

MathOp::Sigmoid_Mat::Sigmoid_Mat(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    res->setAux(new Mat(tmp_r, tmp_c));
    u_st = new Mat(tmp_r, tmp_c);
    u_st_res = new Mat(tmp_r, tmp_c);
    u_nd = new Mat(tmp_r, tmp_c);
    u_nd_res = new Mat(tmp_r, tmp_c);
    pLTZ_f_1 = new LTZ(u_st_res, u_st, BIT_P_LEN);
    pLTZ_f_2 = new LTZ(u_nd_res, u_nd, BIT_P_LEN);
    pDegRed_res = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    pDiv2mP_b1 = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);
    pDiv2mP_b2 = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);

    tmp = new Mat(tmp_r, tmp_c);
    reveal = new Reveal(tmp, u_st_res);
    init(5, 4);
}
//
void MathOp::Sigmoid_Mat::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        *u_st = *a->getForward() + (IE >> 1);
        *u_nd = *a->getForward() - (IE >> 1);
        break;
    case 2:
        pLTZ_f_1->forward();
        if (pLTZ_f_1->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        pLTZ_f_2->forward();
        if (pLTZ_f_2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        *u_st = u_st_res->oneMinus_IE();
        *u_nd = u_nd_res->oneMinus_IE();
        *res->getForward() = (*a->getForward() + (IE >> 1)).dot(*u_st) - (*a->getForward() - (IE >> 1)).dot(*u_nd);
        break;
    case 5:
        pDegRed_res->forward();
        if (pDegRed_res->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::Sigmoid_Mat::back()
{
    backRound++;
    switch (backRound)
    {
    case 1:
        *a->getGrad() = res->getForward()->dot(res->getForward()->oneMinus_IE());
        break;
    case 2:
        pDiv2mP_b1->forward();
        if (pDiv2mP_b1->forwardHasNext())
        {
            backRound--;
        }
        break;
    case 3:
        *a->getGrad() = res->getGrad()->dot(*a->getGrad());
        break;
    case 4:
        pDiv2mP_b2->forward();
        if (pDiv2mP_b2->forwardHasNext())
        {
            backRound--;
        }
        break;
    }
}

MathOp::Argmax::Argmax() {}

MathOp::Argmax::Argmax(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    init(1, 0);
}

void MathOp::Argmax::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        *res->getForward() = a->getForward()->argmax();
        break;
    }
}

void MathOp::Argmax::back() {}

MathOp::Equal::Equal() {}

MathOp::Equal::Equal(NeuronMat *res, NeuronMat *a, NeuronMat *b)
{
    this->res = res;
    this->a = a;
    this->b = b;
    init(1, 0);
}

void MathOp::Equal::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        (*res->getForward())(0, 0) = a->getForward()->eq(*b->getForward()).count();
        break;
    }
}

void MathOp::Equal::back() {}

void MathOp::broadcast(Mat *a)
{
    for (int i = 0; i < M; i++)
    {
        if (i != node_type)
        {
            socket_io[node_type][i]->send_message(a[i]);
        }
    }
}
void MathOp::broadcast_share(Mat *a, int target)
{
    socket_io[node_type][target]->send_message(a);
}

void MathOp::receive_share(Mat *a, int from)
{
    socket_io[node_type][from]->recv_message(*a);
}

void MathOp::broadcase_rep(Mat *a)
{
    for (int i = 0; i < M; i++)
    {
        if (i != node_type)
        {
            socket_io[node_type][i]->send_message(a);
        }
    }
}

void MathOp::receive(Mat *a)
{
    for (int i = 0; i < M; i++)
    {
        if (i != node_type)
        {
            a[i] = socket_io[node_type][i]->recv_message();
        }
    }
}

void MathOp::receive_add(Mat *a)
{
    for (int i = 0; i < M; i++)
    {
        if (i != node_type)
        {
            socket_io[node_type][i]->recv_message(a);
        }
    }
}

void MathOp::receive_rep(Mat *a)
{
    for (int i = 0; i < M; i++)
    {
        if (i != node_type)
        {
            socket_io[node_type][i]->recv_message(*(a + i));
        }
    }
}

void MathOp::random(Mat *a, ll range)
{
    int len = a[0].rows() * a[0].cols();
    ll128 coefficient[TN];
    for (int i = 0; i < len; i++)
    {
        coefficient[0] = (Constant::Util::randomlong() % range);
        for (int j = 1; j < TN; j++)
        {
            coefficient[j] = Constant::Util::randomlong();
        }
        for (int j = 0; j < M; j++)
        {
            ll128 tmp = coefficient[0];
            ll128 key = player[j].key;
            for (int k = 1; k < TN; k++)
            {
                tmp += coefficient[k] * key;
                key *= player[j].key;
                key = Constant::Util::get_residual(key);
            }
            a[j].getVal(i) = tmp;
        }
    }
}

MathOp::Sigmoid::Sigmoid() {}

MathOp::Sigmoid::Sigmoid(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    init();
}

void MathOp::Sigmoid::forward()
{
    reinit();
    *res->getForward() = a->getForward()->sigmoid();
}
void MathOp::Sigmoid::back()
{
    backRound++;
    if (!a->getIsBack())
    {
        *a->getGrad() = res->getForward()->dot(res->getForward()->oneMinus_IE()) / IE;
        *a->getGrad() = a->getGrad()->dot(*res->getGrad()) / IE;
    }
}

MathOp::Hard_Tanh::Hard_Tanh() {}

MathOp::Hard_Tanh::Hard_Tanh(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    init();
}

void MathOp::Hard_Tanh::forward()
{
    reinit();
    *res->getForward() = a->getForward()->hard_tanh();
}
void MathOp::Hard_Tanh::back()
{
    backRound++;
    if (!a->getIsBack())
    {
        *a->getGrad() = res->getForward()->dot(res->getForward()->oneMinus_IE()) / IE;
        *a->getGrad() = a->getGrad()->dot(*res->getGrad()) * 2 / IE;
    }
}

MathOp::Tanh_ex::Tanh_ex() {}

MathOp::Tanh_ex::Tanh_ex(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    init();
}

void MathOp::Tanh_ex::forward()
{
    reinit();
    *res->getForward() = a->getForward()->chebyshev_tanh();
}
void MathOp::Tanh_ex::back()
{
    backRound++;
    if (!a->getIsBack())
    {
        *a->getGrad() = res->getForward()->dot(res->getForward()->oneMinus_IE()) / IE;
        *a->getGrad() = a->getGrad()->dot(*res->getGrad()) * 2 / IE;
    }
}
MathOp::Tanh::Tanh() {}

MathOp::Tanh::Tanh(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    a_r = a->getForward();
    int tmp_r, tmp_c;
    tmp_r = a_r->rows();
    tmp_c = a_r->cols();
    temp_f = new Mat(tmp_r, tmp_c);
    temp_b = new Mat(tmp_r, tmp_c);
    div2mP_f1 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f2 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f4 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f3 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f5 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f6 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f7 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f8 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b1 = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b2 = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);
    init(9, 4);
}

void MathOp::Tanh::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        *res->getForward() = *a_r * coefficients_f[0] / IE;
        break;
    case 2:
        *res->getForward() = (*res->getForward() + coefficients_f[1]).dot(*a_r);
        break;
    case 3:
        div2mP_f1->forward();
        if (div2mP_f1->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        *res->getForward() = (*res->getForward() + coefficients_f[2]).dot(*a_r);
        break;
    case 5:
        div2mP_f2->forward();
        if (div2mP_f2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        *res->getForward() = (*res->getForward() + coefficients_f[3]).dot(*a_r);
        break;
    case 7:
        div2mP_f3->forward();
        if (div2mP_f3->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 8:
        *res->getForward() = (*res->getForward() + coefficients_f[4]).dot(*a_r);
        break;
    case 9:
        div2mP_f4->forward();
        if (div2mP_f4->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::Tanh::back()
{
    backRound++;
    switch (backRound)
    {
    case 1:
    {
        if (!a->getIsBack())
        {
            *a->getGrad() = (*res->getForward() + IE).dot(res->getForward()->oneMinus_IE());
        }
    }
    break;
    case 2:
    {
        if (!a->getIsBack())
        {
            div2mP_b1->forward();
            if (div2mP_b1->forwardHasNext())
            {
                backRound--;
            }
        }
    }
    break;
    case 3:
    {
        if (!a->getIsBack())
        {
            *a->getGrad() = a->getGrad()->dot(*res->getGrad());
        }
    }
    break;
    case 4:
    {
        if (!a->getIsBack())
        {
            div2mP_b2->forward();
            if (div2mP_b2->forwardHasNext())
            {
                backRound--;
            }
        }
    }
    break;
    }
}

MathOp::Tanh_Mat::Tanh_Mat() {}

MathOp::Tanh_Mat::Tanh_Mat(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    a_r = a->getForward();
    div2mP_f1 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f2 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_f3 = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b1 = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b2 = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    res->setAux(new Mat(tmp_r, tmp_c));
    u_st = new Mat(tmp_r, tmp_c);
    u_st_res = new Mat(tmp_r, tmp_c);
    u_nd = new Mat(tmp_r, tmp_c);
    u_nd_res = new Mat(tmp_r, tmp_c);
    reveal = new Reveal(u_nd_res, u_nd);
    u_rd = new Mat(tmp_r, tmp_c);
    u_rd_res = new Mat(tmp_r, tmp_c);
    pLTZ_f_1 = new LTZ(u_st_res, u_st, BIT_P_LEN);
    pLTZ_f_2 = new LTZ(u_rd_res, u_rd, BIT_P_LEN);
    init(9, 4);
}

void MathOp::Tanh_Mat::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        //            *res->getForward() = *a_r * coefficients[0] / IE;
        *res->getForward() = *a_r * coefficients[0] * Constant::Util::inverse(IE, MOD);
        break;
    case 2:
        *res->getForward() = res->getForward()->dot(*a_r);
        break;
    case 3:
        div2mP_f1->forward();
        if (div2mP_f1->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        *res->getForward() = (*res->getForward() + coefficients[1]).dot(*a_r);
        break;
    case 5:
        div2mP_f2->forward();
        if (div2mP_f2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        *u_st = *a->getForward() + IE;
        *u_rd = *a->getForward() - IE;
        break;
    case 7:
        pLTZ_f_1->forward();
        pLTZ_f_2->forward();
        if (pLTZ_f_1->forwardHasNext() || pLTZ_f_2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 8:
        *u_rd_res = u_rd_res->oneMinus_IE();
        *u_nd = (*u_rd_res + *u_st_res).oneMinus_IE();
        *res->getForward() = res->getForward()->dot(*u_nd) + *u_rd_res * IE - *u_st_res * IE;
        break;
    case 9:
        div2mP_f3->forward();
        if (div2mP_f3->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 10:
        *res->getForward() = *res->getForward() + *u_rd_res - *u_st_res;
        break;
    }
}
void MathOp::Tanh_Mat::back()
{
    backRound++;
    switch (backRound)
    {
    case 1:
    {
        if (!a->getIsBack())
        {
            *a->getGrad() = (*res->getForward() + IE).dot(res->getForward()->oneMinus_IE());
        }
    }
    break;
    case 2:
    {
        if (!a->getIsBack())
        {
            div2mP_b1->forward();
            if (div2mP_b1->forwardHasNext())
            {
                backRound--;
            }
        }
    }
    break;
    case 3:
    {
        if (!a->getIsBack())
        {
            *a->getGrad() = a->getGrad()->dot(*res->getGrad());
        }
    }
    break;
    case 4:
    {
        if (!a->getIsBack())
        {
            div2mP_b2->forward();
            if (div2mP_b2->forwardHasNext())
            {
                backRound--;
            }
        }
    }
    break;
    }
}

MathOp::Raw_Tanh::Raw_Tanh() {}

MathOp::Raw_Tanh::Raw_Tanh(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    init();
}

void MathOp::Raw_Tanh::forward()
{
    reinit();
    *res->getForward() = a->getForward()->raw_tanh();
}

void MathOp::Raw_Tanh::back()
{
    backRound++;
    if (!a->getIsBack())
    {
        *a->getGrad() = (*res->getForward() + IE).dot(res->getForward()->oneMinus_IE()) / IE;
        *a->getGrad() = a->getGrad()->dot(*res->getGrad()) / IE;
    }
}

MathOp::SigmaOutput::SigmaOutput() {}

MathOp::SigmaOutput::SigmaOutput(Mat *res, Mat **a, int size)
{
    this->res = res;
    this->a = a;
    this->size = size;
    init(2, 0);
}

void MathOp::SigmaOutput::forward()
{
    reinit();

    switch (forwardRound)
    {
    case 1:
        res->clear();
        break;
    case 2:
        for (int i = 0; i < size; ++i)
        {
            *res += *a[i];
        }
        break;
    }
}

void MathOp::SigmaOutput::back() {}

MathOp::Tanh_change::Tanh_change() {}

MathOp::Tanh_change::Tanh_change(NeuronMat *res, NeuronMat *a)
{
    this->res = res;
    this->a = a;
    init();
}

void MathOp::Tanh_change::forward()
{
    reinit();

    *res->getForward() = (*a->getForward() + IE) * Constant::Util::inverse(2, MOD);
}

void MathOp::Tanh_change::back()
{
    backRound++;
    if (!a->getIsBack())
    {
        *a->getGrad() = *res->getGrad() * 2;
    }
}
/** KV related computations **/
// batch query, not batch kv data
MathOp::SecGetBasic::SecGetBasic() {}

MathOp::SecGetBasic::SecGetBasic(Mat *res, Mat *kv_data, Mat *query_paintext)
{
    this->res = res;
    this->kv_data = kv_data;

    this->query_paintext = query_paintext;

    this->bitmap = new Mat(PREDICTION_BATCH_BENCHMARK, FEATURE_DIM_BENCHMARK);
    this->product = new Mat(PREDICTION_BATCH_BENCHMARK, 1);
    div2mp = new Div2mP(res, product, BIT_P_LEN, DECIMAL_PLACES);

    this->tmp_reveal = new Mat(PREDICTION_BATCH_BENCHMARK, 1);
    reveal = new Reveal(tmp_reveal, res);
    init(4, 0);
}

void MathOp::SecGetBasic::forward()
{
    reinit();

    switch (forwardRound)
    {
    case 1:
    {
        // init
        // dispatch shares of index and val to other parties (clients and commodity server)
        {
            if (node_type == 0)
            {
                Mat input_data(PREDICTION_BATCH_BENCHMARK, FEATURE_DIM_BENCHMARK);
                for (int i = 0; i < input_data.rows(); ++i)
                {
                    int index = query_paintext->getVal(i);
                    for (int j = 0; j < input_data.cols(); ++j)
                    {
                        input_data(i, j) = 0;
                        if (j == index)
                            input_data(i, j) = 1;
                    }
                }

                Mat *share_bitmap = IOManager::secret_share_mat_data(input_data, input_data.size(), "benchmark_secget");

                bitmap = &share_bitmap[0];
                for (int j = 1; j < M; ++j)
                {
                    // share feature
                    MathOp::broadcast_share(&share_bitmap[j], j);
                }
            }
            else
            {
                MathOp::receive_share(bitmap, 0);
            }
        }
        break;
    }
    case 2:
        // scalar product
        *product = (*bitmap) * (*kv_data); // Nx1 matrix
        break;
    case 3:
        div2mp->forward();
        if (div2mp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        break;
    case 5:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        cout << "Reveal Feature\n";
        tmp_reveal->print();
        break;
    }
}

void MathOp::SecGetBasic::back() {}

MathOp::SecGetFold::SecGetFold() {}

MathOp::SecGetFold::SecGetFold(Mat *res, Mat *kv_data, Mat *query_paintext)
{
    this->fold_dim = ceil(sqrt(SECGET_DIM));

    this->res = res;
    this->kv_data = kv_data;
    this->kv_data_reshape = new Mat(fold_dim, fold_dim);
    this->query_paintext = query_paintext;
    if (isMarkovInference)
        query_num = SECGET_BATCH * MAX_LEN;
    else
        query_num = SECGET_BATCH;

    this->bitmap = new Mat(query_num * 2, fold_dim);

    this->row = new Mat(query_num, fold_dim);
    this->row_trunc = new Mat(query_num, fold_dim);
    this->product = new Mat(query_num, 1);
    div2mp_row = new Div2mP(row_trunc, row, BIT_P_LEN, DECIMAL_PLACES);
    div2mp_res = new Div2mP(res, product, BIT_P_LEN, DECIMAL_PLACES);

    this->tmp_reveal = new Mat(query_num, 1);
    reveal = new Reveal(tmp_reveal, res);
    cout << "fold dim: " << fold_dim << endl;
    init(5, 0);
}

void MathOp::SecGetFold::forward()
{
    reinit();
    // cout << "SecGetFold: " << forwardRound <<endl;feature_dim
    switch (forwardRound)
    {
    case 1:
    {
        // init
        // dispatch shares of index and val to other parties (clients and commodity server)
        Mat dummy(1, 1);
        {
            if (node_type == 0)
            {

                // for Markov model, with each password of MAX_LEN
                Mat input_data(query_num * 2, fold_dim);
                if (isMarkovInference)
                {
                    for (int i = 0; i < SECGET_BATCH; i++)
                    {
                        /* code */
                        for (int j = 0; j < MAX_LEN; j++)
                        {
                            int index = query_paintext->get(i, j);
                            int dim_x = index / fold_dim;
                            int dim_y = index % fold_dim;
                            input_data(i * MAX_LEN + j, dim_x) = 1;
                            input_data(i * MAX_LEN + j + query_num, dim_y) = 1;
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < SECGET_BATCH; ++i)
                    {
                        int index = query_paintext->getVal(i);
                        int dim_x = index / fold_dim;
                        int dim_y = index % fold_dim;
                        input_data(i, dim_x) = 1;
                        input_data(i + SECGET_BATCH, dim_y) = 1;
                    }
                }

                cout << "Send size: " << input_data.size() << endl;
                Mat *share_bitmap = IOManager::secret_share_mat_data(input_data, input_data.size(), "benchmark_secget");
                bitmap = &share_bitmap[0];

                for (int j = 1; j < M; ++j)
                {
                    // share feature
                    MathOp::broadcast_share(&share_bitmap[j], j);
                    MathOp::receive_share(&dummy, j);
                }
            }
            else
            {
                MathOp::broadcast_share(&dummy, 0); //
                MathOp::receive_share(bitmap, 0);
            }
        }
        bitmap_x = bitmap->row(0, query_num);
        bitmap_y = bitmap->row(query_num, query_num * 2);
        break;
    }
    case 2:
    {
        // reshape kv_data: (D, 1) -> (\sqrt D, \sqrt D)
        // matrix mul
        // kv_data->print();
        // pad
        int new_size = fold_dim * fold_dim;
        *kv_data = kv_data->resize(new_size, 1);
        *kv_data_reshape = kv_data->resize(fold_dim, fold_dim);
        *row = bitmap_x * (*kv_data_reshape); // Nx1 matrix
        break;
    }
    case 3:
        div2mp_row->forward();
        if (div2mp_row->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        // *res = *row_trunc;
        // Mat tmp_product = std::move();
        *product = bitmap_y.dot(*row_trunc).reduce_sum_x();
        break;
    case 5:
        div2mp_res->forward();
        if (div2mp_res->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 7:
        cout << "Reveal Res\n";
        tmp_reveal->print();
        break;
    }
}

void MathOp::SecGetFold::back() {}

MathOp::SecGetOnline::SecGetOnline() {}

MathOp::SecGetOnline::SecGetOnline(Mat *res, Mat *kv_data, Mat *query_paintext, Mat *perm_mat, Mat *perm_mat_plain)
{
    this->res = res;
    this->kv_data = kv_data;
    this->kv_data_shuffle = new Mat(kv_data->rows(), kv_data->cols());
    this->query_paintext = query_paintext;

    this->perm_mat = perm_mat;
    this->perm_mat_plain = perm_mat_plain;

    div2mp_data = new Div2mP(kv_data_shuffle, kv_data_shuffle, BIT_P_LEN, DECIMAL_PLACES);

    this->tmp_reveal = new Mat(kv_data->rows(), kv_data->cols());

    init(6, 0);
}

void MathOp::SecGetOnline::forward()
{
    reinit();
    // cout << "SecGetOnline: " << forwardRound <<endl;
    switch (forwardRound)
    {
    // sample random permutation matrix, if the feed input: perm_mat and perm_mat_plain is not empty, this case 1 shall not be performed
    case 1:
    {
        break;
    }
    case 2:
        break;
    case 3:
    {
        // *kv_data_shuffle = *perm_mat * *kv_data;
        *kv_data_shuffle = *kv_data; // without pre-computation
        break;
    }
    case 4:
        break;
    case 5:
    {
        // *res = *kv_data_shuffle;
        // break;
        // dispatch shares of index and val to other parties (clients and commodity server)
        Mat dummy(1, 1);
        {
            if (node_type == 0)
            {
                // init
                for (int i = 0; i < query_paintext->rows(); i++)
                {
                    int target_index = query_paintext->getVal(i);
                    for (int j = 0; j < perm_mat_plain->rows(); j++)
                    {
                        if (perm_mat_plain->get(j, target_index) == 1)
                        {
                            query_paintext->setVal(i, j);
                            break;
                        }
                    }
                }
                for (int j = 1; j < M; ++j)
                {
                    // share feature
                    MathOp::broadcast_share(query_paintext, j);
                    MathOp::receive_share(&dummy, j);
                }
            }
            else
            {
                MathOp::broadcast_share(&dummy, 0); //
                MathOp::receive_share(query_paintext, 0);
            }
        }
        // if (node_type != 1)
        //     cout << "Feature in: " << socket_io[node_type][1]->send_num << endl;
        break;
    }
    case 6:
        // reshape kv_data: (D, 1) -> (\sqrt D, \sqrt D)
        // matrix mul
        for (int i = 0; i < query_paintext->rows(); i++)
        {
            res->setVal(i, kv_data_shuffle->getVal(query_paintext->getVal(i)));
        }
        query_paintext->clear();
        break;
    }
}

void MathOp::SecGetOnline::back()
{
}

MathOp::SecGetPer::SecGetPer(Mat *res, Mat *kv_data, Mat *index_mat, int shuffle_type)
{
    this->res = res;
    this->kv_data = kv_data;
    this->index_mat = index_mat;
    this->shuffle_type = shuffle_type;
    res->init(index_mat->rows(), index_mat->cols());

    kv_r = kv_data->rows();
    // shuffe_kv = new Mat(kv_r, kv_data->cols());

    switch (shuffle_type)
    {
    case 0:
        shuffledata = new MathOp::PShuffleData(kv_data, kv_data);
        break;
    case 1:
        shuffledata = new MathOp::EShuffleData(kv_data, kv_data);
        break;
    case 2:
        shuffledata = new MathOp::ShuffleDataNoGP(kv_data, kv_data);
        break;
    default:
        shuffledata = new MathOp::EShuffleData(kv_data, kv_data);
        break;
    }
    init(3, 0);
}

void MathOp::SecGetPer::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        if (OFFLINE_PHASE_ON)
        {
            shuffledata->forward();
            if (shuffledata->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 2:
    { // shuffle index
      // In fact, this step should be accomplished by client(client know permutation rule)
      // To mock step of shuffle index, P0 act as client.
        Mat dummy(1, 1);
        {
            if (node_type == 0)
            {
                shuffledata->ShuffleIndex(index_mat); // FIXME: offline
                for (int j = 1; j < M; ++j)
                {
                    // share index after shuffle
                    MathOp::broadcast_share(index_mat, j);
                    MathOp::receive_share(&dummy, j);
                }
            }
            else
            {
                MathOp::broadcast_share(&dummy, 0); //
                MathOp::receive_share(index_mat, 0);
            }
        }
        break;
    }
    case 3:
        for (int i = 0; i < index_mat->cols(); i++)
        {
            for (int j = 0; j < index_mat->rows(); j++)
            {
                // row,col,value
                // query index: index_mat->get(j, i)
                // value: shuffle_kv[query index,c]
                res->setVal(j, i, kv_data->get(index_mat->get(j, i), i));
            }
        }
        break;
    }
}

void MathOp::SecGetPer::back()
{
}

MathOp::GenPerm::GenPerm() {}

MathOp::GenPerm::GenPerm(Mat *pr_mat, Mat *r_mat)
{

    this->r_mat = r_mat;
    this->pr_mat = pr_mat;
    r = r_mat->rows();
    c = r_mat->cols();

    pRandFld = new PRandFld(r_mat, MOD);

    permObjNext = new PermutationObj(r, c);
    permObjPrev = new PermutationObj(r, c);

    reshare0 = new ReShare(pr_mat, pr_mat, 0, 1);
    reshare1 = new ReShare(pr_mat, pr_mat, 1, 2);
    reshare2 = new ReShare(pr_mat, pr_mat, 2, 0);

    init(8, 0);
}

// n0[pie_next] * r
// n1[pie_prev] * r
// n0[pie_next] ==  n1[pie_prev] for they generate from same k0
void MathOp::GenPerm::piemulr(int n0, int n1)
{
    if (node_type == n0)
    {
        permObjNext->PermMat(pr_mat);
    }
    else if (node_type == n1)
    {
        permObjPrev->PermMat(pr_mat);
    }
}

// only work for three-party
void MathOp::GenPerm::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        // generate random shared vector r_mat
        pRandFld->forward();
        if (pRandFld->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    // get result pr_mat
    // (pie2,pie0) (pie0,pie1)  (pie1,pie2)
    // p0 p1
    case 2:
        *pr_mat = *r_mat;

        // get permutationObj
        player[node_type].rand_prev->initPermMat(permObjPrev);
        player[node_type].rand_next->initPermMat(permObjNext);
        break;
    case 3:
        // pie0 * r
        piemulr(0, 1);
        break;
    case 4:
        reshare0->forward();
        if (reshare0->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
        // pie1 * r
        piemulr(1, 2);
        break;
    case 6:
        reshare1->forward();
        if (reshare1->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 7:
        piemulr(2, 0);
        break;
    case 8:
        reshare2->forward();
        if (reshare2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

PermutationObj *MathOp::GenPerm::getPermObjNext()
{
    return permObjNext;
}

PermutationObj *MathOp::GenPerm::getPermObjPrev()
{
    return permObjPrev;
}

void MathOp::GenPerm::GenPerm::back() {}

MathOp::ShuffleData::ShuffleData(Mat *res, Mat *kv_data)
{
    this->kv_data = kv_data;
    this->res = res;

    r = kv_data->rows();
    c = kv_data->cols();

    r_mat = new Mat(r, c);
    pr_mat = new Mat(r, c);

    p_xr = new Mat(r, c);
    p_xr_plain = new Mat(r, c);

    x_r = new Mat(r, c);
}

MathOp::ShuffleData::ShuffleData() {}

MathOp::ShuffleData::~ShuffleData() {}

MathOp::PShuffleData::PShuffleData(Mat *res, Mat *kv_data) : ShuffleData(res, kv_data)
{
    xr_plain = new Mat(r, c);
    xr_reveal = new Reveal(xr_plain, x_r);

    genperm = new GenPerm(pr_mat, r_mat);

    permExtra = new PermutationObj(r, c);

    init(6, 0);
}

void MathOp::PShuffleData::forward()
{
    reinit();
    switch (forwardRound)
    {
    // offline
    case 1: // generate random permutation pair
        if (OFFLINE_PHASE_ON)
        {
            genperm->forward();
            if (genperm->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 2:
        if (OFFLINE_PHASE_ON)
        {
            // p0 has an extra permutation
            if (node_type == 0)
            {
                player[node_type].rand_extra->initPermMat(permExtra);
            }
        }
        break;
    // shuffle the key-value data
    case 3: // kv_data - random (x-r)
        *x_r = *kv_data + *r_mat;
        break;
    case 4: // reveal x-r
        xr_reveal->forward();
        if (xr_reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
    {
        Mat dummy(1, 1);
        // p*(x+r) and send ss of p*(x+r) to other
        if (node_type == 0)
        {
            // (x+r)*pie0*pie1*pie2
            *p_xr_plain = *xr_plain;
            genperm->getPermObjNext()->PermMat(p_xr_plain);
            permExtra->PermMat(p_xr_plain);
            genperm->getPermObjPrev()->PermMat(p_xr_plain);

            *p_xr = *p_xr_plain;
            for (int j = 1; j < M; ++j)
            {
                // share feature
                MathOp::broadcast_share(p_xr_plain, j);
                MathOp::receive_share(&dummy, j);
            }
        }
        else
        {
            MathOp::broadcast_share(&dummy, 0); //
            MathOp::receive_share(p_xr, 0);
        }
        break;
    }
    case 6: // p*(x+r) - p*(r)
        *res = *p_xr - *pr_mat;
        break;
    }
}

void MathOp::PShuffleData::ShuffleIndex(Mat *index_mat)
{
    if (node_type == 0)
    {
        genperm->getPermObjNext()->PermIndex(index_mat);
        permExtra->PermIndex(index_mat);
        genperm->getPermObjPrev()->PermIndex(index_mat);
    }
}

void MathOp::PShuffleData::back() {}

MathOp::EShuffleData::EShuffleData(Mat *res, Mat *kv_data) : ShuffleData(res, kv_data)
{
    genperm = new GenPerm(pr_mat, r_mat);

    permExtra = new PermutationObj(r, c); // for query party
    init(5, 0);
}

void MathOp::EShuffleData::forward()
{
    reinit();
    switch (forwardRound)
    {
    // offline
    case 1: // generate random permutation pair
        if (OFFLINE_PHASE_ON)
        {
            genperm->forward();
            if (genperm->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 2:
        if (OFFLINE_PHASE_ON)
        {
            // p0 has an extra permutation
            if (node_type == 0)
            {
                player[node_type].rand_extra->initPermMat(permExtra);
            }
        }
        break;
    // offline
    // shuffle the key-value data
    case 3:
    {
        *x_r = *kv_data + *r_mat;
        *p_xr_plain = *x_r;
        if (node_type == 0 || node_type == 1)
        {
            if (node_type == 0)
            {
                genperm->getPermObjNext()->PermMat(p_xr_plain);
                *p_xr_plain = *p_xr_plain * player[node_type].reshare_key_next;
            }
            else
            {
                genperm->getPermObjPrev()->PermMat(p_xr_plain);
                *p_xr_plain = *p_xr_plain * player[node_type].reshare_key_prev;
            }

            // distribute share
            socket_io[node_type][2]->send_message(p_xr_plain);
        }
        else
        {
            // <p_xr_plain>0 + <p_xr_plain>1
            MathOp::receive_share(p_xr_plain, 0);
            socket_io[node_type][1]->recv_message(p_xr_plain);
        }
        break;
    }
    case 4:
    {
        Mat dummy(1, 1);
        if (node_type == 2)
        {
            // pie2*pie1 * (x-r) == p_xr_plain
            genperm->getPermObjPrev()->PermMat(p_xr_plain);
            genperm->getPermObjNext()->PermMat(p_xr_plain);

            // share p_xr_plain
            *p_xr = *p_xr_plain;
            for (int j = 0; j < 2; ++j)
            {
                // share index after shuffle
                MathOp::broadcast_share(p_xr_plain, j);
                MathOp::receive_share(&dummy, j);
            }
        }
        else
        {
            MathOp::broadcast_share(&dummy, 2);
            MathOp::receive_share(p_xr, 2);
        }
        break;
    }
    case 5:
        *res = *p_xr - *pr_mat;
        break;
    default:
        break;
    }
}

void MathOp::EShuffleData::ShuffleIndex(Mat *index_mat)
{
    if (node_type == 0)
    {
        genperm->getPermObjNext()->PermIndex(index_mat);
        permExtra->PermIndex(index_mat);
        genperm->getPermObjPrev()->PermIndex(index_mat);
    }
}

void MathOp::EShuffleData::back()
{
}

MathOp::ShuffleDataNoGP::ShuffleDataNoGP(Mat *res, Mat *kv_data) : ShuffleData(res, kv_data)
{
    pRandFld = new PRandFld(r_mat, MOD);

    p_mat_plain = new Mat(r, r);
    p_mat = new Mat(r, r);

    div2mp = new Div2mP(pr_mat, pr_mat, BIT_P_LEN, DECIMAL_PLACES);

    xr_plain = new Mat(r, c);
    xr_reveal = new Reveal(xr_plain, x_r);

    r_plain = new Mat(r, c);
    pr_plain = new Mat(r, c);
    r_reveal = new Reveal(r_plain, r_mat);
    pr_reveal = new Reveal(pr_plain, pr_mat);

    res_reveal = new Reveal(res, res);

    init(7, 0);
}

void MathOp::ShuffleDataNoGP::forward()
{
    reinit();
    switch (forwardRound)
    {
    // offline
    case 1:
        // generate random mat
        if (OFFLINE_PHASE_ON)
        {
            pRandFld->forward();
            if (pRandFld->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 2:
        // local p*r
        // ss permutation mat
        if (OFFLINE_PHASE_ON)
        {
            if (node_type == 0) // for p0, own the plain permutation mat
            {
                //  generate permutation mat
                *p_mat_plain = Mat::random_permutation_matrix(r, r);

                Mat *share_perm_mat = IOManager::secret_share_mat_data(*p_mat_plain, p_mat_plain->size(), "secget perm mat");
                *p_mat = share_perm_mat[0];
                for (int j = 1; j < M; ++j)
                {
                    // share feature
                    MathOp::broadcast_share(&share_perm_mat[j], j);
                }
            }
            else
            {
                MathOp::receive_share(p_mat, 0);
            }
            // perm the random mat
            *pr_mat = *p_mat * *r_mat;
        }
        break;
    case 3: // p*r
        if (OFFLINE_PHASE_ON)
        {
            div2mp->forward();
            if (div2mp->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    // online
    case 4:
        // x-r
        *x_r = *kv_data - *r_mat;
        break;
    case 5:
        // reveal x-r
        xr_reveal->forward();
        if (xr_reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        // p*(x-r)
        if (node_type == 0)
        {
            *p_xr_plain = *p_mat_plain * *xr_plain;
            *p_xr = *p_xr_plain;
            for (int j = 1; j < M; ++j)
            {
                MathOp::broadcast_share(p_xr_plain, j);
            }
        }
        else
        {
            MathOp::receive_share(p_xr, 0);
        }
        break;
    case 7:
        // p(x-r) + p*r
        *res = *p_xr + *pr_mat;
        break;
    }
}

void MathOp::ShuffleDataNoGP::ShuffleIndex(Mat *index_mat)
{
    if (node_type == 0)
    {
        for (int i = 0; i < index_mat->size(); i++)
        {
            int temp = index_mat->getVal(i);

            // find new index
            for (int j = 0; j < r; j++)
            {
                if (p_mat_plain->get(j, temp))
                {
                    index_mat->setVal(i, j);
                    break;
                }
            }
        }
    }
}

void MathOp::ShuffleDataNoGP::back() {}

/** Implementation of decision tree **/
MathOp::DT_entropy::DT_entropy() {}

MathOp::DT_entropy::DT_entropy(NodeMat *node)
{
    this->node = node;
    init(0, 0);
}

void MathOp::DT_entropy::forward(ll *res)
{
    *res = 0;
}

MathOp::entropy_gain::entropy_gain() {}

MathOp::entropy_gain::entropy_gain(NodeMat *node)
{
    this->node = node;
    init(0, 0);
}

void MathOp::entropy_gain::forward(ll *res)
{
    *res = 0;
}

MathOp::feature_split_finding::feature_split_finding() {}

MathOp::feature_split_finding::feature_split_finding(NodeMat *node)
{
    this->node = node;
    init(0, 0);
}

void MathOp::feature_split_finding::forward(ll *res)
{
    *res = 0;
}

/** Decision tree inference operators**/
MathOp::FeatureNode::FeatureNode() {}

MathOp::FeatureNode::FeatureNode(Mat *res, Mat *table_data, Mat *a, int feature_index)
{
    this->feature_index = feature_index;
    this->table_data = table_data;
    this->a = a;
    this->res = res;

    this->bitmap = new Mat(FEATURE_DIM, 1);
    this->product = new Mat(PREDICTION_BATCH, 1);
    this->product_deg = new Mat(PREDICTION_BATCH, 1);

    div2mp = new Div2mP(res, product, BIT_P_LEN, DECIMAL_PLACES);


    this->tmp_reveal = new Mat(PREDICTION_BATCH, 1);
    reveal = new Reveal(tmp_reveal, res);
    init(5, 0);

    // permute data (SecGet Online)
    Constant::Clock clock = Constant::Clock::get_clock(10);
    this->perm_mat = new Mat(FEATURE_DIM, FEATURE_DIM);
    this->perm_mat_plain = new Mat(FEATURE_DIM, FEATURE_DIM);
    this->table_data_shuffle = new Mat(table_data->rows(), table_data->cols());
    this->query_plaintex = new Mat(1, 1);
    div2mp_shuffle = new Div2mP(table_data_shuffle, table_data_shuffle, BIT_P_LEN, DECIMAL_PLACES);

    DBGprint("send perm tot_time: %.3f\n", clock.get());
}

void MathOp::FeatureNode::forward()
{
    reinit();

    switch (forwardRound)
    {
    // Online: with permutation offline
    case 1:
    {
        // *table_data_shuffle = *table_data * *perm_mat;
        *table_data_shuffle = *table_data;
        break;
    }
    case 2:
        break;
    case 3:
        if (node_type == 0)
        {
            // FIXME: not shuffle
            query_plaintex->setVal(0, feature_index);
            for (int j = 1; j < M; ++j)
            {
                // share feature
                MathOp::broadcast_share(query_plaintex, j);
            }
        }
        else
        {
            MathOp::receive_share(query_plaintex, 0);
        }
        for (int i = 0; i < res->rows(); i++)
        {
            res->setVal(i, table_data_shuffle->get(i, query_plaintex->get(0, 0)));
        }
        break;
    // Raw porotocol, use SecGet Basic Bitmap
    case 4:
        break;
    case 5:
        break;
    case 6:
        break;
    }
}

void MathOp::FeatureNode::back() {}

MathOp::Internal::Internal() {}

MathOp::Internal::Internal(Mat *res, Mat *vector, double feature_val, int sign, Mat *test, int sibling, Mat *sibling_test)
{
    this->vector = vector;
    this->res = res;
    this->feature_val = feature_val;
    this->val_mat = new Mat(PREDICTION_BATCH, 1);
    this->tmp = new Mat(PREDICTION_BATCH, 1);
    this->cur_test = new Mat(PREDICTION_BATCH, 1);
    this->test = test;
    this->accumulate_product = new Mat(PREDICTION_BATCH, 1);
    //    this->eqz = new EQZ_2LTZ(res, tmp, BIT_P_LEN);
    this->sign = sign;
    if (sign == 0)
    {
        this->eqz = new EQZ(cur_test, tmp, BIT_P_LEN);
    }
    else if (sign == 1)
    {
        this->eqz = new LTZ(cur_test, tmp, BIT_P_LEN);
    }
    else if (sign == 2)
    {
        // GT, x > thr
        this->eqz = new LTZ(cur_test, tmp, BIT_P_LEN);
    }
    div2mp = new Div2mP(res, accumulate_product, BIT_P_LEN, DECIMAL_PLACES);

    this->tmp_reveal = new Mat(PREDICTION_BATCH, 1);
    reveal = new Reveal(tmp_reveal, res);
    init(7, 0);

    this->sibling = sibling;
    this->sibling_test = sibling_test;

    // dispatch shares of val to other parties (clients and commodity server)
    double *tmp_data = new double[1];
    tmp_data[0] = feature_val;
    Mat *share_val = IOManager::secret_share_vector(tmp_data, 1);
    if (node_type == 0)
    {
        val_mat = &share_val[0];
        shared_val = val_mat->getVal(0);
        for (int j = 1; j < M; ++j)
        {
            // share feature
            MathOp::broadcast_share(&share_val[j], j);
        }
    }
    else
    {
        MathOp::receive_share(val_mat, 0);
        shared_val = val_mat->getVal(0);
    }
}

void MathOp::Internal::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
    {
        if (sibling > 0)
        {
            *this->cur_test = sibling_test->oneMinus_IE();
            this->forwardRound = 4;
        }
        break;
    }
    case 2:
        // eqz test
        //            vector->print();
        *tmp = *vector - shared_val;
        break;
    case 3:
        eqz->forward();
        if (eqz->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        if (sign == 2)
        {
            // GT
            *cur_test = cur_test->oneMinus();
        }
        *cur_test = *cur_test * IE;
        // cout << "val: " << shared_val << endl;
        // if (node_type != 1)
        //     cout << "Internal "<< sign << " out: " << socket_io[node_type][1]->send_num << endl;
        break;
    case 5:
        // cout << "accumulate produuct\n";
        *accumulate_product = (*cur_test).dot(*test); // Nx1 matrix
        break;
    case 6:
        div2mp->forward();
        if (div2mp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 7:
        break;
    }
}

void MathOp::Internal::back() {}

// Decision Tree Permutation
MathOp::TreePerm::TreePerm() {}

MathOp::TreePerm::TreePerm(NodeMat *res, Mat *feature_vector, Mat *threshold)
{
    this->res = res;

    // node parameter
    this->feature_vector = feature_vector;
    this->threshold = threshold;

    // randomness provided by PRF
    shuffle_idxs = new Mat[NODE_NUM];
    shuffle_thresholds = new Mat[NODE_NUM];
    reshare_idxs1 = new ReShare *[NODE_NUM];
    reshare_thresholds1 = new ReShare *[NODE_NUM];
    reshare_idxs2 = new ReShare *[NODE_NUM];
    reshare_thresholds2 = new ReShare *[NODE_NUM];
    reshare_idxs3 = new ReShare *[NODE_NUM];
    reshare_thresholds3 = new ReShare *[NODE_NUM];

    tmp_bits = new Mat(1, DEPTH - 1);
    first_bits = new Mat(1, DEPTH - 1);
    second_bits = new Mat(1, DEPTH - 1);
    final_bits = new Mat(1, DEPTH - 1);
    final_bits_reveal = new Mat(1, DEPTH - 1);
    degRed = new DegRed(final_bits, final_bits);
    reveal_final_bits = new Reveal(final_bits_reveal, final_bits);

    reveal1 = new Mat[NODE_NUM];
    reveal2 = new Mat[NODE_NUM];
    reveal_idxs = new Reveal *[NODE_NUM];
    reveal_thresholds = new Reveal *[NODE_NUM];

    for (int i = 0; i < NODE_NUM; i++)
    {
        shuffle_idxs[i].init(1, FEATURE_VECTOR_SIZE);
        shuffle_thresholds[i].init(1, 1);
        // first permute
        reshare_idxs1[i] = new ReShare(&feature_vector[i], &shuffle_idxs[i], 0, 1);
        reshare_thresholds1[i] = new ReShare(&threshold[i], &shuffle_thresholds[i], 0, 1);
        // second permute
        reshare_idxs2[i] = new ReShare(&feature_vector[i], &shuffle_idxs[i], 1, 2);
        reshare_thresholds2[i] = new ReShare(&threshold[i], &shuffle_thresholds[i], 1, 2);

        // third permute
        reshare_idxs3[i] = new ReShare(&feature_vector[i], &shuffle_idxs[i], 2, 0);
        reshare_thresholds3[i] = new ReShare(&threshold[i], &shuffle_thresholds[i], 2, 0);

        reveal1[i].init(1, FEATURE_VECTOR_SIZE);
        reveal2[i].init(1, 1);
        reveal_idxs[i] = new Reveal(&reveal1[i], &feature_vector[i]);
        reveal_thresholds[i] = new Reveal(&reveal2[i], &threshold[i]);
    }

    init(19, 0);
}

void MathOp::TreePerm::forward()
{
    reinit();
    // cout << "TreePerm: " << forwardRound << endl;
    switch (forwardRound)
    {
    case 1:
    {
        if (LOG_MESSAGES)
        {
            cout << "feature vector\n";
            for (int i = 0; i < NODE_NUM; i++)
            {
                feature_vector[i].print();
                threshold[i].print();
            }
        }

        std::pair<std::vector<int>, std::vector<int>> res;
        // p0 and p1 permute
        if (node_type == 0 || node_type == 1)
        {
            if (node_type == 0)
            {
                res = player[0].rand_next->getTreePermutation(DEPTH);
                perm_bits_nxt = res.first;
            }
            else
            {
                res = player[1].rand_prev->getTreePermutation(DEPTH);
                perm_bits_prev = res.first;
            }
            tree_permutation = res.second;
            cout << "Tree Permutation P0 and P1\n";
            for (int i : tree_permutation)
                cout << i << ", ";
            cout << endl;

            // firstly permuted tree.
            for (int i = 0; i < NODE_NUM; i++)
            {
                for (int j = 0; j < FEATURE_VECTOR_SIZE; j++)
                {
                    shuffle_idxs[i].setVal(0, j, feature_vector[tree_permutation[i] - 1].get(0, j));
                }
                shuffle_thresholds[i].setVal(0, 0, threshold[tree_permutation[i] - 1].get(0, 0));
            }

            if (LOG_MESSAGES)
            {
                cout << "Before ReShare\n";
                for (int i = 0; i < NODE_NUM; i++)
                {
                    shuffle_idxs[i].print();
                    shuffle_thresholds[i].print();
                }
            }
        }

        break;
    }
    case 2:
    {
        // reshare feature and threshold
        for (int i = 0; i < NODE_NUM; i++)
        {
            reshare_idxs1[i]->forward();
            reshare_thresholds1[i]->forward();
        }
        for (int i = 0; i < NODE_NUM; i++)
        {
            if (reshare_idxs1[i]->forwardHasNext() || reshare_thresholds1[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    }
    case 3:
        break;
    case 4:
    {
        if (LOG_MESSAGES)
        {
            cout << "First Permute Res\n";
            for (int i = 0; i < NODE_NUM; i++)
            {
                reveal1[i].print();
                reveal2[i].print();
            }
        }
        break;
    }
    case 5:
    {
        if (LOG_MESSAGES)
        {
            cout << "feature vector\n";
            for (int i = 0; i < NODE_NUM; i++)
            {
                feature_vector[i].print();
                threshold[i].print();
            }
        }

        std::pair<std::vector<int>, std::vector<int>> res;
        // p0 and p1 permute
        if (node_type == 1 || node_type == 2)
        {
            if (node_type == 1)
            {
                res = player[1].rand_next->getTreePermutation(DEPTH);
                perm_bits_nxt = res.first;
            }
            else
            {
                res = player[2].rand_prev->getTreePermutation(DEPTH);
                perm_bits_prev = res.first;
            }
            tree_permutation = res.second;
            cout << "Tree Permutation P1 and P2\n";
            for (int i : tree_permutation)
                cout << i << ", ";
            cout << endl;

            // second permuted tree.
            for (int i = 0; i < NODE_NUM; i++)
            {
                for (int j = 0; j < FEATURE_VECTOR_SIZE; j++)
                {
                    shuffle_idxs[i].setVal(0, j, feature_vector[tree_permutation[i] - 1].get(0, j));
                }
                shuffle_thresholds[i].setVal(0, 0, threshold[tree_permutation[i] - 1].get(0, 0));
            }
            if (LOG_MESSAGES)
            {
                cout << "Before ReShare\n";
                for (int i = 0; i < NODE_NUM; i++)
                {
                    shuffle_idxs[i].print();
                    shuffle_thresholds[i].print();
                }
            }
        }

        break;
    }
    case 6:
    {
        // reshare feature and threshold
        for (int i = 0; i < NODE_NUM; i++)
        {
            reshare_idxs2[i]->forward();
            reshare_thresholds2[i]->forward();
        }
        for (int i = 0; i < NODE_NUM; i++)
        {
            if (reshare_idxs2[i]->forwardHasNext() || reshare_thresholds2[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    }
    case 7:
        break;
    case 8:
    {
        if (LOG_MESSAGES)
        {
            cout << "Second Permute Res\n";
            for (int i = 0; i < NODE_NUM; i++)
            {
                reveal1[i].print();
                reveal2[i].print();
            }
        }
        break;
    }
    case 9:
    {
        if (LOG_MESSAGES)
        {
            cout << "feature vector\n";
            for (int i = 0; i < NODE_NUM; i++)
            {
                feature_vector[i].print();
                threshold[i].print();
            }
        }

        std::pair<std::vector<int>, std::vector<int>> res;
        // p2 and p0 permute
        if (node_type == 2 || node_type == 0)
        {
            if (node_type == 2)
            {
                res = player[2].rand_next->getTreePermutation(DEPTH);
                perm_bits_nxt = res.first;
            }
            else
            {
                res = player[0].rand_prev->getTreePermutation(DEPTH);
                perm_bits_prev = res.first;
            }
            tree_permutation = res.second;
            cout << "Tree Permutation P2 and P0\n";
            for (int i : tree_permutation)
                cout << i << ", ";
            cout << endl;

            // third permuted tree.
            for (int i = 0; i < NODE_NUM; i++)
            {
                for (int j = 0; j < FEATURE_VECTOR_SIZE; j++)
                {
                    shuffle_idxs[i].setVal(0, j, feature_vector[tree_permutation[i] - 1].get(0, j));
                }
                shuffle_thresholds[i].setVal(0, 0, threshold[tree_permutation[i] - 1].get(0, 0));
            }

            if (LOG_MESSAGES)
            {
                cout << "Before ReShare\n";
                for (int i = 0; i < NODE_NUM; i++)
                {
                    shuffle_idxs[i].print();
                    shuffle_thresholds[i].print();
                }
            }
        }
        break;
    }
    case 10:
    {
        // reshare feature and threshold
        for (int i = 0; i < NODE_NUM; i++)
        {
            reshare_idxs3[i]->forward();
            reshare_thresholds3[i]->forward();
        }
        for (int i = 0; i < NODE_NUM; i++)
        {
            if (reshare_idxs3[i]->forwardHasNext() || reshare_thresholds3[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    }
    case 11:
        // reveal feature and threshold
        if (LOG_MESSAGES)
        {
            for (int i = 0; i < NODE_NUM; i++)
            {
                reveal_idxs[i]->forward();
                reveal_thresholds[i]->forward();
            }
            for (int i = 0; i < NODE_NUM; i++)
            {
                if (reveal_idxs[i]->forwardHasNext() || reveal_thresholds[i]->forwardHasNext())
                {
                    forwardRound--;
                    break;
                }
            }
        }
        break;
    case 12:
    {
        if (LOG_MESSAGES)
        {
            cout << "Third Permute Res\n";
            for (int i = 0; i < NODE_NUM; i++)
            {
                reveal1[i].print();
                reveal2[i].print();
            }
        }
        break;
    }
    case 13:
    {
        // obtain the joint perm bits
        if (node_type == 0 || node_type == 1)
        {
            if (node_type == 0)
            {
                for (int i = 0; i < DEPTH - 1; i++)
                {
                    tmp_bits->setVal(0, i, perm_bits_nxt[i] ^ perm_bits_prev[i]);
                }
            }
            else if (node_type == 1)
            {
                for (int i = 0; i < DEPTH - 1; i++)
                {
                    tmp_bits->setVal(0, i, perm_bits_nxt[i]);
                }
            }
            // tmp_bits->print();
        }
        break;
    }
    case 14:
    {
        if (node_type == 0)
        {
            Mat *shares = IOManager::secret_share_mat_data(*tmp_bits, tmp_bits->size(), "");
            *first_bits = shares[0];
            // distribute share
            for (int i = 0; i < M; i++)
            {
                if (i != node_type)
                {
                    socket_io[node_type][i]->send_message(shares[i]);
                }
            }
        }
        else
        {
            MathOp::receive_share(first_bits, 0);
        }
        break;
    }
    case 15:
    {
        if (node_type == 1)
        {
            Mat *shares = IOManager::secret_share_mat_data(*tmp_bits, tmp_bits->size(), "");
            *second_bits = shares[1];
            // distribute share
            for (int i = 0; i < M; i++)
            {
                if (i != node_type)
                {
                    socket_io[node_type][i]->send_message(shares[i]);
                }
            }
        }
        else
        {
            MathOp::receive_share(second_bits, 1);
        }
        break;
    }
    case 16:
    {
        Mat tmp_prod = first_bits->dot(*second_bits);
        *final_bits = *first_bits + *second_bits - tmp_prod * 2;
        break;
    }
    case 17:
        degRed->forward();
        if (degRed->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 18:
        break;
    case 19:
    {
        // final_bits_reveal->print();
        // assign reture values
        Mat *perm_bits = new Mat[DEPTH - 1];
        for (int i = 0; i < DEPTH - 1; i++)
        {
            perm_bits[i].init(1, 1);
            perm_bits[i].setVal(0, 0, final_bits_reveal->get(0, i));
        }
        res->set_idxs(feature_vector);
        res->set_thresholds(threshold);
        res->set_perm_bits(perm_bits);
        break;
    }
    }
}

void MathOp::TreePerm::back() {}

MathOp::OptInternalNode::OptInternalNode() {}

MathOp::OptInternalNode::OptInternalNode(Mat *res, Mat *input, Mat *feature_vector, Mat *threshold, Mat *perm_bit)
{
    this->res = res;
    this->input = input;

    // node parameter
    this->feature_vector = feature_vector;
    this->threshold = threshold;
    this->perm_bit = perm_bit;

    // fold based feature selection
    if (!PERMUTE_USING_BASIC)
    {
        this->fold_num = feature_vector->cols() / 2; // 2 * dim
        this->vector_x = new Mat(1, fold_num);
        this->vector_y = new Mat(PREDICTION_BATCH, fold_num);
        this->rows = new Mat(PREDICTION_BATCH, fold_num);

        div2mp_row = new Div2mP(rows, rows, BIT_P_LEN, DECIMAL_PLACES);
    }

    // selected feature value
    this->feature_val = new Mat(PREDICTION_BATCH, 1);

    // comparison
    this->cur_test = new Mat(PREDICTION_BATCH, 1);
    this->comparison_res = new Mat(PREDICTION_BATCH, 1);

    this->ltz = new LTZ(comparison_res, cur_test, BIT_P_LEN);

    // final left or right after XOR com_res with perm_bit
    this->xor_res = new Mat(PREDICTION_BATCH, 1);

    this->feature_vec_reveal = new Mat(1, FEATURE_DIM);
    this->product = new Mat(PREDICTION_BATCH, 1);
    this->com_res_reveal = new Mat(PREDICTION_BATCH, 1);
    this->feature_val_reveal = new Mat(PREDICTION_BATCH, 1);
    this->res_reveal = new Mat(PREDICTION_BATCH, 1);

    reveal_res = new Reveal(res, xor_res);
    reveal_feature = new Reveal(feature_vec_reveal, feature_vector);
    reveal_feature_val = new Reveal(feature_val_reveal, feature_val);
    reveal_com_res = new Reveal(com_res_reveal, comparison_res);
    div2mp = new Div2mP(feature_val, product, BIT_P_LEN, DECIMAL_PLACES);

    init(16, 0);
}

void MathOp::OptInternalNode::forward()
{
    reinit();
    // cout << "Opt Internal Node: " << forwardRound << endl;
    switch (forwardRound)
    {
    case 1:
    {
        // feature selection
        if (LOG_MESSAGES)
        {
            reveal_feature->forward();
            if (reveal_feature->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    }
    case 2:
        if (LOG_MESSAGES)
        {
            cout << "feature vector\n";
            feature_vec_reveal->print();
        }
        break;
    case 3:
        // NxD * NxD , Basic-impl
        if (PERMUTE_USING_BASIC)
        {
            *product = input->dot(*feature_vector).reduce_sum_x();
            forwardRound = 6; // jump pass the fold-impl
        }
        break;
    case 4:
    {
        Mat tmp(fold_num, fold_num);
        start = system_clock::now();
        // reshape kv-data: NxD --> N x \sqrt D x \sqrt D
        for (int idx = 0; idx < PREDICTION_BATCH; idx++)
        {
            // for i in N:
            //      N x \sqrt D * \sqrt D x \sqrt D
            for (int i = 0; i < fold_num; i++)
            {
                for (int j = 0; j < fold_num; j++)
                {
                    if (i * fold_num + j < FEATURE_DIM)
                        tmp.setVal(j * fold_num + i, input->get(idx, i * fold_num + j));
                }
            }

            for (int j = 0; j < fold_num; j++)
            {
                vector_x->setVal(0, j, feature_vector->get(idx, j));
                vector_y->setVal(idx, j, feature_vector->get(idx, j + fold_num));
            }
            Mat row = *vector_x * tmp;
            for (int i = 0; i < fold_num; i++)
            {
                rows->setVal(idx, i, row.getVal(i));
            }
        }
        end = system_clock::now();
        time_span = duration_cast<microseconds>(end - start);
        break;
    }
    case 5:
        div2mp_row->forward();
        if (div2mp_row->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        end = system_clock::now();
        time_span = duration_cast<microseconds>(end - start);
        // NxD * NxD , Basic-impl
        *product = rows->dot(*vector_y).reduce_sum_x();
        break;
    case 7:
        div2mp->forward();
        if (div2mp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 8:
        if (LOG_MESSAGES)
        {
            reveal_feature_val->forward();
            if (reveal_feature_val->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 9:
        if (LOG_MESSAGES)
        {
            cout << "val: " << endl;
            feature_val_reveal->print();
        }
        break;
    case 10:
        // comparison
        if (LOG_MESSAGES)
        {
            cout << "product\n";
            product->print();
        }
        *cur_test = *product - *threshold;
        break;
    case 11:
        ltz->forward();
        if (ltz->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 12:
        if (LOG_MESSAGES)
        {
            reveal_com_res->forward();
            if (reveal_com_res->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 13:
        if (LOG_MESSAGES)
        {
            cout << "com res\n";
            com_res_reveal->print();
        }
        break;
    case 14:
    {
        // comparison_res \in {0, 1}
        // XOR perm_bit
        Mat tmp_prod = comparison_res->dot(*perm_bit);
        *xor_res = *comparison_res + *perm_bit - tmp_prod * 2;
        break;
    }
    case 15:
        reveal_res->forward();
        if (reveal_res->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 16:
        if (LOG_MESSAGES)
        {
            cout << "final res: \n";
            res->print();
        }
        break;
    }
}

void MathOp::OptInternalNode::back() {}

MathOp::Leaf::Leaf() {}

MathOp::Leaf::Leaf(Mat *res, Mat *eqz_res, int label)
{
    this->label = label;
    this->eqz_res = eqz_res;
    this->res = res;

    this->label_mat = new Mat(1, 1);
    this->product = new Mat(PREDICTION_BATCH, 1);
    div2mp = new Div2mP(res, product, BIT_P_LEN, DECIMAL_PLACES);

    int r = eqz_res->rows();
    tmp_reveal = new Mat(r, 1);
    reveal = new Reveal(tmp_reveal, eqz_res);
    init(3, 0);

    {
        // dispatch shares of val to other parties (clients and commodity server)
        double *tmp_data = new double[1];
        tmp_data[0] = label;
        Mat *share_label = IOManager::secret_share_vector(tmp_data, 1);
        if (node_type == 0)
        {
            label_mat = &share_label[0];
            shared_label = label_mat->getVal(0);
            for (int j = 1; j < M; ++j)
            {
                // share feature
                MathOp::broadcast_share(&share_label[j], j);
            }
        }
        else
        {
            MathOp::receive_share(label_mat, 0);
            shared_label = label_mat->getVal(0);
        }
    }
}

void MathOp::Leaf::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
    {
        break;
    }
    case 2:
        // tmp_reveal->print();
        *product = (*eqz_res) * shared_label;
        break;
    case 3:
        div2mp->forward();
        if (div2mp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::Leaf::back() {}

/** Implementation of MAP functions **/
MathOp::Map_Get::Map_Get() {}

MathOp::Map_Get::Map_Get(Mat *node, Mat *a, Mat *b)
{
    this->node = node;
    this->a = a;
    this->b = b;
}

void MathOp::Map_Get::forward()
{
}

void MathOp::Map_Get::back() {}

MathOp::Map_Set::Map_Set() {}

MathOp::Map_Set::Map_Set(Mat *node, Mat *a, Mat *b)
{
}

void MathOp::Map_Set::forward()
{
}

void MathOp::Map_Set::back() {}

/** Implementation of complex non-linear functions **/
/** Premise: k > 2 **/
/** Following the idea from https://github.com/LaRiffle/approximate-models **/
MathOp::Pow_log::Pow_log() {}

MathOp::Pow_log::Pow_log(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;

    int row = a->rows();
    int col = a->cols();
    this->base = new Mat(row, col);
    this->tmp = new Mat(row, col);

    this->div2mp_tmp = new Div2mP(tmp, tmp, BIT_P_LEN, DECIMAL_PLACES);
    this->div2mp_res = new Div2mP(res, res, BIT_P_LEN, DECIMAL_PLACES);
    this->perform_div2mp = false;

    init(3, 0);
}

void MathOp::Pow_log::forward()
{
    reinit();
    //    cout << "Exp: " << forwardRound <<endl;
    switch (forwardRound)
    {
    case 1:
    {
        *tmp = *a;
        cur_k = k;
        for (int i = 0; i < res->size(); ++i)
        {
            res->setVal(i, IE);
        }
        perform_div2mp = false;
        break;
    }
    case 2:
    {
        // when b == 0, jump out the case
        if (cur_k == 0)
        {
            break;
        }
        if (!perform_div2mp)
        {
            if (cur_k & 1)
            {
                *res = res->dot(*tmp);
                div2mp_res->reset_for_multi_call(); // reset operators according to reinit
                                                    //                    globalRound++;
                                                    //                    localRound++;
            }
            *tmp = tmp->dot(*tmp);
            div2mp_tmp->reset_for_multi_call();
            last_k = cur_k;
            cur_k = cur_k >> 1;
            perform_div2mp = true;
            forwardRound--;
        }
        else
        {
            if (last_k & 1)
            {
                div2mp_res->forward();
                div2mp_tmp->forward();
                if (div2mp_tmp->forwardHasNext() || div2mp_res->forwardHasNext())
                {
                    forwardRound--;
                }
                else
                {
                    perform_div2mp = false;
                    forwardRound--;
                }
            }
            else
            {
                //                    cout << "bbb\n";
                div2mp_tmp->forward();
                if (div2mp_tmp->forwardHasNext())
                {
                    forwardRound--;
                }
                else
                {
                    perform_div2mp = false;
                    forwardRound--;
                }
            }
        }
        break;
    }
    case 3:
        div2mp_res->forward();
        if (div2mp_res->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::Pow_log::back() {}

MathOp::Exp_approximate::Exp_approximate() {}

MathOp::Exp_approximate::Exp_approximate(Mat *res, Mat *a, int exp_iterations)
{
    this->res = res;
    this->a = a;
    this->exp_iterations = exp_iterations;

    int row = a->rows();
    int col = a->cols();
    this->base = new Mat(row, col);

    this->cur = new Mat(row, col);
    this->base_tr = new Mat(row, col);

    this->div2mp = new Div2mP(base_tr, base, BIT_P_LEN, DECIMAL_PLACES);
    this->pow_log = new Pow_log(res, base_tr, pow(2, exp_iterations));

    this->tmp1 = new Mat(row, col);
    this->tmp2 = new Mat(row, col);
    this->reveal = new Reveal(tmp1, base);
    this->reveal_tmp = new Reveal(tmp2, base_tr);

    init(6, 0);
}

void MathOp::Exp_approximate::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
    {
        ll n = pow(2, exp_iterations);
        cout << "n: " << n << endl;
        *base = *a * (IE / n);
        break;
    }
    case 2:
        div2mp->forward();
        if (div2mp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        *base_tr = *base_tr + IE;
        break;
    case 4:
        pow_log->forward();
        if (pow_log->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
        reveal->forward();
        reveal_tmp->forward();
        if (reveal->forwardHasNext() || reveal_tmp->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 6:
        cout << "base: \n";
        tmp1->print();
        cout << "base_tr: \n";
        tmp2->print();
        break;
    }
}

void MathOp::Exp_approximate::back() {}

MathOp::Log_approximate::Log_approximate() {}

MathOp::Log_approximate::Log_approximate(Mat *res, Mat *a, int k, int f)
{
    this->res = res;
    this->a = a;
    this->k = k;
    this->f = f;
    this->a_B = new Mat[k];
    this->c_B = new Mat[k];
    this->x_pie = new Mat[k];
    this->mul_a_B = new Mat[k];
    this->div2mp_a_B = new Div2mP *[k];
    int row = a->rows();
    int col = a->cols();

    for (int i = 0; i < k; i++)
    {
        a_B[i].init(row, col);
        c_B[i].init(row, col);
        x_pie[i].init(row, col);
        mul_a_B[i].init(row, col);
        div2mp_a_B[i] = new Div2mP(mul_a_B + i, mul_a_B + i, k, f);
    }

    this->exp = new Mat(row, col);
    this->b = new Mat(row, col);
    this->x_st = new Mat(row, col);
    this->exp_res = new Mat(row, col);
    this->xb = new Mat(row, col);
    this->Div2mp_1 = new Div2mP(exp_res, exp_res, k, f);
    this->Div2mp_2 = new Div2mP(exp_res, exp_res, k, f);
    this->bitDec = new BitDec(a_B, a, k, k);
    this->sufOrC = new SufOrC(c_B, a_B, k - 1);
    this->pDiv2mP = new Div2mP(x_st, xb, k, k - f - 12);
    this->divIE = new Div2mP(exp_res, exp_res, k, f);

    int depth = 3;

    init(18, 0);
}

void MathOp::Log_approximate::forward()
{
    reinit();

    switch (forwardRound)
    {
    case 1:
        bitDec->forward();
        if (bitDec->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        break;
    case 3:
        break;
    case 4:
        break;
    case 5:
        break;
    case 6:
        sufOrC->forward();
        if (sufOrC->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 7:
        break;
    case 8:
        exp->clear();
        for (int i = 0; i < k - 1; i++)
        {
            *exp = *exp + c_B[i];
        }
        *exp = *exp - f - 1;
        break;
    case 9:
        b->clear();
        for (int i = 0; i < k - 1; i++)
        {
            *b += c_B[i].oneMinus() * (1ll << (k - i - 2));
        }
        *b = *b + 1;
        *b = *b * Constant::Util::inverse(1 << 10, MOD); // in order to avoid overflow
        // return the approximation of log_2(x) = log_2(x') + k
        *xb = b->dot(*a);
        break;
    case 10:
        pDiv2mP->forward();
        if (pDiv2mP->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 11:
        // compute Appro
        *exp_res = *x_st * coeff_degrees[depth - 1][0];
        break;
    case 12:
        divIE->forward();
        if (divIE->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 13:
        if (depth < 2)
        {
            forwardRound = 16;
            break;
        }
        *exp_res = *exp_res + coeff_degrees[depth - 1][1];
        *exp_res = exp_res->dot(*x_st);
        break;
    case 14:
        Div2mp_1->forward();
        if (Div2mp_1->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 15:
        if (depth < 3)
        {
            forwardRound = 16;
            break;
        }
        *exp_res = *exp_res + coeff_degrees[depth - 1][2];
        *exp_res = exp_res->dot(*x_st);
        break;
    case 16:
        Div2mp_2->forward();
        if (Div2mp_2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 17:
        *exp_res = *exp_res + coeff_degrees[depth - 1][3];
        *res = *exp_res + (*exp * IE);
        break;
    case 18:
        for (int i = 0; i < k - 1; i++)
        {
            c_B[i].clear();
        }
        break;
    default:
        break;
    }
}

void MathOp::Log_approximate::back() {}

MathOp::SufMulInv::SufMulInv() {}
MathOp::SufMulInv::SufMulInv(Mat *p_B, Mat *p_inv_B, Mat *d_B, int k)
{
    this->p_B = p_B;
    this->p_inv_B = p_inv_B;
    this->d_B = d_B;
    this->k = k;
    int tmp_r = d_B[0].rows();
    int tmp_c = d_B[0].cols();
    c_B = new Mat[k];
    c_rev_B = new Mat[k];
    c_inv_rev_B = new Mat[k];
    r_B = new Mat[k];
    p_inv_rev_B = new Mat[k];
    p_rev_B = new Mat[k];
    d_rev_B = new Mat[k];
    d_reveal_B = new Mat[k];
    test_1 = new Mat[k];
    test_2 = new Mat[k];
    test_3 = new Mat[k];

    for (int i = 0; i < k; i++)
    {
        c_B[i].init(tmp_r, tmp_c);
        c_rev_B[i].init(tmp_r, tmp_c);
        c_inv_rev_B[i].init(tmp_r, tmp_c);
        r_B[i].init(tmp_r, tmp_c);
        p_inv_rev_B[i].init(tmp_r, tmp_c);
        p_rev_B[i].init(tmp_r, tmp_c);
        d_rev_B[i].init(tmp_r, tmp_c);
        d_reveal_B[i].init(tmp_r, tmp_c);
        test_1[i].init(tmp_r, tmp_c);
        test_2[i].init(tmp_r, tmp_c);
        test_3[i].init(tmp_r, tmp_c);
    }

    this->preMulC = new PreMulC(p_rev_B, d_rev_B, k);
    this->prandFld_c = new PRandFld *[k];
    this->div2mp1 = new Div2mP *[k];
    this->div2mp2 = new Div2mP *[k];
    this->reveal = new Reveal *[k];
    this->reveal_d = new Reveal *[k];
    this->reveal_test1 = new Reveal *[k];
    this->reveal_test2 = new Reveal *[k];
    this->reveal_test3 = new Reveal *[k];
    this->mulpub = new MulPub *[k];
    for (int i = 0; i < k; i++)
    {
        prandFld_c[i] = new PRandFld(r_B + i, MOD);
        div2mp1[i] = new Div2mP(c_B + i, c_B + i, BIT_P_LEN, DECIMAL_PLACES);
        div2mp2[i] = new Div2mP(p_inv_rev_B + i, p_inv_rev_B + i, BIT_P_LEN, DECIMAL_PLACES);
        reveal[i] = new Reveal(c_rev_B + i, c_B + i);
        reveal_d[i] = new Reveal(d_reveal_B + i, d_B + i);
        reveal_test1[i] = new Reveal(test_1 + i, p_B + i);
        reveal_test2[i] = new Reveal(test_2 + i, p_inv_B + i);
        reveal_test3[i] = new Reveal(test_3 + i, r_B + i);
        mulpub[i] = new MulPub(c_rev_B + i, r_B + i, p_rev_B + i);
    }

    init(9, 0);
}
void MathOp::SufMulInv::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        for (int i = 0; i < k; i++)
        {
            d_rev_B[i] = d_B[k - 1 - i];
        }
        /* code */
        break;

    case 2:
        preMulC->forward();
        if (preMulC->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            for (int i = 0; i < k; i++)
            {
                prandFld_c[i]->forward();
            }
            for (int i = 0; i < k; i++)
            {
                if (prandFld_c[i]->forwardHasNext())
                {
                    forwardRound--;
                    break;
                }
            }
        }
        break;
    case 4:
        for (int i = 0; i < k; i++)
        {
            c_B[i] = r_B[i].dot(p_rev_B[i]);
        }
        break;
    case 5:
        break;
    case 6:
        for (int i = 0; i < k; i++)
        {
            reveal[i]->forward();
        }
        for (int i = 0; i < k; i++)
        {
            if (reveal[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 7:
        for (int i = 0; i < k; i++)
        {
            c_inv_rev_B[i] = c_rev_B[i].inverse();
            // * --> dot, multiply r_B
            p_inv_rev_B[i] = c_inv_rev_B[i].dot(r_B[i]);
        }
        break;
    case 8:
        break;
    case 9:
        for (int i = 0; i < k; i++)
        {
            p_B[i] = p_rev_B[k - 1 - i];
            p_inv_B[i] = p_inv_rev_B[k - 1 - i];
        }
        break;
    default:
        break;
    }
}
void MathOp::SufMulInv::back() {}

MathOp::PreBitLT::PreBitLT() {}
MathOp::PreBitLT::PreBitLT(Mat *res, Mat *a, Mat *b_B, int k)
{
    this->res = res;
    this->a = a;
    this->b_B = b_B;
    this->k = k;
    int tmp_r = a->rows();
    int tmp_c = a->cols();
    p_B = new Mat[k];
    p_st_B = new Mat[k];
    d_B = new Mat[k];
    d_B_plus_one = new Mat[k];
    s_B = new Mat[k];
    tmp_mul_sp = new Mat[k];
    test_1 = new Mat(tmp_r, tmp_c);
    test_2 = new Mat[k];
    test_3 = new Mat[k];
    test_4 = new Mat(tmp_r, tmp_c);
    test_5 = new Mat(tmp_r, tmp_c);

    for (int i = 0; i < k; i++)
    {
        p_B[i].init(tmp_r, tmp_c);
        p_st_B[i].init(tmp_r, tmp_c);
        d_B[i].init(tmp_r, tmp_c);
        d_B_plus_one[i].init(tmp_r, tmp_c);
        s_B[i].init(tmp_r, tmp_c);
        tmp_mul_sp[i].init(tmp_r, tmp_c);
        test_2[i].init(tmp_r, tmp_c);
        test_3[i].init(tmp_r, tmp_c);
    }
    sufMulInv = new SufMulInv(p_B, p_st_B, d_B_plus_one, k);
    mod2D = new Mod2 *[k];
    div2mp = new Div2mP *[k];

    reveal_test1 = new Reveal(test_1, a);
    reveal_test2 = new Reveal *[k];
    reveal_test3 = new Reveal *[k];
    reveal_test4 = new Reveal(test_4, res);
    reveal_test5 = new Reveal(test_5, res + 1);
    for (int i = 0; i < k - 1; i++)
    {
        mod2D[i] = new Mod2(&res[i], &tmp_mul_sp[i], k);
        div2mp[i] = new Div2mP(&tmp_mul_sp[i], &tmp_mul_sp[i], BIT_P_LEN, DECIMAL_PLACES);
    }
    for (int i = 0; i < k; i++)
    {
        reveal_test2[i] = new Reveal(test_2 + i, b_B + i);
        reveal_test3[i] = new Reveal(test_3 + i, res + i);
    }
    mod2 = new Mod2(&res[k - 1], &s_B[k - 1], k);
    init(8, 0);
};
void MathOp::PreBitLT::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        for (int i = 0; i < k; i++)
        {
            d_B[i] = a->get_bit(i) + b_B[i] - (a->get_bit(i) * 2).dot(b_B[i]);
            d_B[i].residual();
        }
        break;
    case 2:
        for (int i = 0; i < k; i++)
        {
            d_B_plus_one[i] = d_B[i] + 1;
        }
        break;
    case 3:
        sufMulInv->forward();
        if (sufMulInv->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 4:
        s_B[0] = a->get_bit(0).oneMinus().dot(p_B[0] - p_B[1]);
        for (int i = 1; i < k - 1; i++)
            s_B[i] = s_B[i - 1] + a->get_bit(i).oneMinus().dot(p_B[i] - p_B[i + 1]);
        s_B[k - 1] = s_B[k - 2] + a->get_bit(k - 1).oneMinus().dot(d_B[k - 1]);
        break;
    case 5:
        for (int i = 0; i < k - 1; i++)
        {
            tmp_mul_sp[i] = s_B[i].dot(p_st_B[i + 1]);
        }
        break;
    case 6:
        break;
    case 7:
        for (int i = 0; i < k - 1; i++)
        {
            mod2D[i]->forward();
        }
        for (int i = 0; i < k - 1; i++)
        {
            if (mod2D[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 8:
        mod2->forward();
        if (mod2->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    default:
        break;
    }
}

void MathOp::PreBitLT::back() {}

MathOp::BitDec::BitDec() {}

MathOp::BitDec::BitDec(Mat *res, Mat *a, int k, int m)
{
    this->res = res;
    this->a = a;
    this->k = k;
    this->m = m;
    int tmp_r = a->rows();
    int tmp_c = a->cols();

    this->b_temp = new Mat[m];
    for (int i = 0; i < m; i++)
    {
        b_temp[i].init(tmp_r, tmp_c);
    }
    this->a_reveal = new Mat(tmp_r, tmp_c);
    this->reveal = new Reveal *[m];
    this->reveal_a = new Reveal(a_reveal, a);
    this->reveal_b_temp = new Mat[m];
    for (int i = 0; i < m; i++)
    {
        reveal_b_temp[i].init(tmp_r, tmp_c);
        reveal[i] = new Reveal(reveal_b_temp + i, res + i);
    }

    this->preMod2m = new PreMod2m(b_temp, a, k, m);
    init(2, 0);
}

void MathOp::BitDec::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        preMod2m->forward();
        if (preMod2m->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 2:
        res[0] = b_temp[0];
        for (int i = 1; i < m; i++)
            res[i] = (b_temp[i] - b_temp[i - 1]) * Constant::Util::inverse(1ll << (i), MOD); // modified
        break;
    default:
        break;
    }
}
void MathOp::BitDec::back() {}

MathOp::PreMod2m::PreMod2m() {}

MathOp::PreMod2m::PreMod2m(Mat *res, Mat *a, int k, int m)
{
    this->res = res;
    this->a = a;
    this->k = k;
    this->m = m;
    int tmp_r = a->rows();
    int tmp_c = a->cols();
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    temp_c = new Mat(tmp_r, tmp_c);
    temp_r = new Mat(tmp_r, tmp_c);
    a_reveal = new Mat(tmp_r, tmp_c);
    temp_c_dec = new Mat[m];
    temp_s_dec = new Mat[m];
    temp_u_dec = new Mat[m];
    res_reveal = new Mat[m];
    r_B = new Mat[m];
    for (int i = 0; i < m; i++)
    {
        temp_c_dec[i].init(tmp_r, tmp_c);
        temp_s_dec[i].init(tmp_r, tmp_c);
        temp_u_dec[i].init(tmp_r, tmp_c);
        res_reveal[i].init(tmp_r, tmp_c);
        r_B[i].init(tmp_r, tmp_c);
    }
    r = new Mat(tmp_r, tmp_c);
    b = new Mat(tmp_r, tmp_c);
    pRandM = new PRandM(r_nd, r_st, r_B, k, m);
    reveal = new Reveal(temp_c, temp_r);
    reveal_a = new Reveal(a_reveal, a);
    reveal_res = new Reveal *[m];
    for (int i = 0; i < m; i++)
    {
        reveal_res[i] = new Reveal(res_reveal + i, res + i);
    }
    preBitLT = new PreBitLT(temp_u_dec, temp_c, r_B, k);
    init(12, 0);
}

void MathOp::PreMod2m::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        r_nd->clear();
        r_st->clear();
        for (int i = 0; i < m; i++)
        {
            r_B[i].clear();
        }
        r->clear();
        b->clear();
        break;
    case 2:
        // todo: offline
        if (OFFLINE_PHASE_ON)
        {
            pRandM->forward();
            if (pRandM->forwardHasNext())
            {
                forwardRound--;
            }
        }
        break;
    case 3:
        *temp_r = (*r_nd) * (1ll << (m)) + (*r_st) + (*a) + (1ll << (k - 1)); // modified
        break;
    case 4:
        reveal->forward();
        if (reveal->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 5:
        for (int i = 0; i < m; i++)
        {
            temp_c_dec[i] = temp_c->mod(1ll << (i + 1));
            temp_s_dec[i].clear();
            for (int j = 0; j < i + 1; j++)
            {
                temp_s_dec[i] += r_B[j] * (1ll << (j));
            }
        }
        break;
    case 6:
        preBitLT->forward();
        if (preBitLT->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 7:
        for (int i = 0; i < m; i++)
            res[i] = temp_c_dec[i] - temp_s_dec[i] + temp_u_dec[i] * (1ll << (i + 1)); // modified
        break;
    default:
        break;
    }
}

void MathOp::PreMod2m::back() {}

MathOp::SufOrC::SufOrC() {}

MathOp::SufOrC::SufOrC(Mat *res, Mat *a, int k)
{
    this->res = res;
    this->a = a;
    this->k = k;
    int tmp_r = a[0].rows();
    int tmp_c = a[0].cols();
    a_rev = new Mat[k];
    res_rev = new Mat[k];
    res_reveal = new Mat[k];
    b = new Mat[k];
    for (int i = 0; i < k; i++)
    {
        a_rev[i].init(tmp_r, tmp_c);
        res_rev[i].init(tmp_r, tmp_c);
        res_reveal[i].init(tmp_r, tmp_c);
        b[i].init(tmp_r, tmp_c);
    }
    preMulC = new PreMulC(b, a_rev, k);
    mod2 = new Mod2 *[k];
    reveal_res = new Reveal *[k];
    for (int i = 0; i < k; i++)
    {
        mod2[i] = new Mod2(res_rev + i, b + i, k);
        reveal_res[i] = new Reveal(res_reveal + i, res + i);
    }
    init(4, 0);
}
void MathOp::SufOrC::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        for (int i = 0; i < k; i++)
        {
            a_rev[i] = a[k - 1 - i] + 1;
        }
        res_rev[0] = a_rev[0] - 1;
        break;
    case 2:
        preMulC->forward();
        if (preMulC->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    case 3:
        for (int i = 1; i < k; i++)
        {
            mod2[i]->forward();
        }
        for (int i = 1; i < k; i++)
        {
            if (mod2[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 4:
        for (int i = 1; i < k; i++)
        {
            res_rev[i] = res_rev[i].oneMinus();
        }
        for (int i = 0; i < k; i++)
        {
            res[i] = res_rev[k - i - 1];
        }
        break;
    case 5:
        for (int i = 0; i < k; i++)
        {
            reveal_res[i]->forward();
        }
        for (int i = 0; i < k; i++)
        {
            if (reveal_res[i]->forwardHasNext())
            {
                forwardRound--;
                break;
            }
        }
        break;
    case 6:
        for (int i = 0; i < k; i++)
        {
            cout << i << endl;
            res_reveal[i].print();
        }
        break;
    default:
        break;
    }
}
void MathOp::SufOrC::back() {}

MathOp::DivMat_CTO::DivMat_CTO() {}

MathOp::DivMat_CTO::DivMat_CTO(Mat *res, Mat *a, Mat *b)
{
    this->res = res;
    this->a = a;
    int tmp_c_a = a->cols();
    int tmp_r_b = b->rows();
    this->division = new Mat(tmp_r_b, tmp_c_a, 0);
    this->exponent = new int[tmp_r_b];
}

void MathOp::DivMat_CTO::forward()
{
    reinit();
    switch (forwardRound)
    {
    case 1:
        // init
        *res = a->dot(*b);
        break;
    case 2:
        div2mP->forward();
        if (div2mP->forwardHasNext())
        {
            forwardRound--;
        }
        break;
    }
}

void MathOp::DivMat_CTO::back() {}