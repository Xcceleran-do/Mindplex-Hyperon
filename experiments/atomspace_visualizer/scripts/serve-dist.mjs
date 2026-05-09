import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';

const port = Number(process.env.PORT || 3001);
const host = process.env.HOST || '127.0.0.1';
const apiTarget = new URL(process.env.API_PROXY_TARGET || 'http://127.0.0.1:5000');
const distRoot = resolve(process.cwd(), 'dist');

const mimeTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.ico', 'image/x-icon'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.map', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.txt', 'text/plain; charset=utf-8'],
]);

const send = (response, status, body, headers = {}) => {
  response.writeHead(status, headers);
  response.end(body);
};

const proxyApi = (request, response) => {
  const targetUrl = new URL(request.url || '/', apiTarget);
  const proxyRequest = fetch(targetUrl, {
    method: request.method,
    headers: request.headers,
    body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request,
    duplex: 'half',
  });

  proxyRequest
    .then(async (proxyResponse) => {
      response.writeHead(proxyResponse.status, Object.fromEntries(proxyResponse.headers));
      if (!proxyResponse.body) {
        response.end();
        return;
      }
      for await (const chunk of proxyResponse.body) {
        response.write(chunk);
      }
      response.end();
    })
    .catch((error) => {
      send(response, 502, JSON.stringify({ error: `API proxy failed: ${error.message}` }), {
        'content-type': 'application/json; charset=utf-8',
      });
    });
};

const resolveStaticPath = (urlPath) => {
  const safePath = normalize(decodeURIComponent(urlPath.split('?')[0] || '/')).replace(/^(\.\.[/\\])+/, '');
  const requestedPath = resolve(distRoot, `.${safePath}`);

  if (!requestedPath.startsWith(distRoot)) {
    return null;
  }

  if (existsSync(requestedPath) && statSync(requestedPath).isFile()) {
    return requestedPath;
  }

  return join(distRoot, 'index.html');
};

const server = createServer((request, response) => {
  if ((request.url || '').startsWith('/api/')) {
    proxyApi(request, response);
    return;
  }

  const filePath = resolveStaticPath(request.url || '/');
  if (!filePath || !existsSync(filePath)) {
    send(response, 404, 'Not found', { 'content-type': 'text/plain; charset=utf-8' });
    return;
  }

  const extension = extname(filePath);
  const cacheControl = filePath.includes(`${join('dist', 'assets')}`) ? 'public, max-age=3600' : 'no-cache';
  response.writeHead(200, {
    'cache-control': cacheControl,
    'content-type': mimeTypes.get(extension) || 'application/octet-stream',
  });
  createReadStream(filePath).pipe(response);
});

server.listen(port, host, () => {
  console.log(`Mindplex frontend preview running at http://${host}:${port}`);
  console.log(`Proxying /api to ${apiTarget.origin}`);
});
