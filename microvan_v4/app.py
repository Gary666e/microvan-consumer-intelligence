"""Launch with python microvan_v4/app.py; open http://127.0.0.1:8765."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from app_core import score,optimize,monitor,summary,COLS,P
class Handler(BaseHTTPRequestHandler):
 def respond(self,data,status=200,mime='application/json'):
  b=data.encode() if isinstance(data,str) else json.dumps(data,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):
  if self.path=='/':self.respond((P/'app.html').read_text(),mime='text/html')
  elif self.path=='/columns':self.respond(COLS)
  elif self.path=='/summary':self.respond(summary())
  else:self.respond({'error':'Not found'},404)
 def do_POST(self):
  try:
   b=json.loads(self.rfile.read(int(self.headers['Content-Length'])));fun={'/score':lambda:score(b['values']),'/optimize':lambda:optimize(b['budget'],b['capacity'],b['floor']),'/monitor':lambda:monitor(b['batch'])};self.respond(fun[self.path]())
  except Exception as e:self.respond({'error':str(e)},400)
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
