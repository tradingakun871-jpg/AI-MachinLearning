import json

from app.database import database_health, latest_training_state, market_history_status


def main():
    payload={
        "database":database_health(),
        "xau_history":market_history_status("XAUUSD"),
        "training":{
            "XAUUSD":latest_training_state("XAUUSD"),
            "BTCUSD":latest_training_state("BTCUSD"),
        },
    }
    print("HISTORY_PROBE="+json.dumps(payload,default=str),flush=True)


if __name__=="__main__":
    main()
