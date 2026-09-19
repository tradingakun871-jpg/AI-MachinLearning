from ml.high_winrate_v123 import _directional_stability


def fold(n,buy=None,sell=None):
    return {
        "fold":n,
        "sides":{
            "BUY":{"policy_mode":"TEST","trading":buy or {"trades":0}},
            "SELL":{"policy_mode":"TEST","trading":sell or {"trades":0}},
        },
    }


def trade(trades,wr,exp,pf,dd=-5.0):
    return {
        "trades":trades,
        "win_rate":wr,
        "expectancy_r":exp,
        "profit_factor":pf,
        "max_drawdown_r":dd,
    }


def main():
    observed={
        "folds":[
            fold(1,sell=trade(182,.32967,-.01099,.9836,-29)),
            fold(2,sell=trade(113,.33628,.00885,1.0133,-18)),
            fold(3,sell=trade(228,.27632,-.17105,.7636,-39)),
            fold(4,buy=trade(165,.38788,.16364,1.2673,-12),sell=trade(19,.36842,.10526,1.1667,-3)),
        ]
    }
    current=_directional_stability(observed,2.0)
    assert current["BUY"]["passed"] is False,current
    assert current["SELL"]["passed"] is False,current

    stable={
        "folds":[
            fold(1,buy=trade(60,.40,.20,1.35)),
            fold(2,buy=trade(70,.39,.17,1.28)),
            fold(3,buy=trade(65,.38,.14,1.22)),
            fold(4,buy=trade(80,.41,.23,1.40)),
        ]
    }
    positive=_directional_stability(stable,2.0)
    assert positive["BUY"]["passed"] is True,positive
    print("V0125_DIRECTIONAL_GATE_OK")
    print({"observed":current,"stable_control":positive})


if __name__=="__main__":
    main()
