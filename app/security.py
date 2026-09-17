import os,hmac
from fastapi import Header,HTTPException

def verify_bridge_token(x_bridge_token:str|None=Header(default=None)):
    expected=os.getenv("BRIDGE_TOKEN","")
    if not expected or expected=="change-me": raise HTTPException(503,"bridge token is not configured")
    if not x_bridge_token or not hmac.compare_digest(x_bridge_token,expected): raise HTTPException(401,"invalid bridge token")
    return True
