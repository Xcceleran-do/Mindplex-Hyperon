import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig(({ mode }) => {
  const apiProxyTarget = mode === 'development' ? 'http://localhost:5000' : undefined;

  return {
    plugins: [solidPlugin()],
    server: {
      port: 3000,
      host: '0.0.0.0',
      proxy: apiProxyTarget
        ? {
            '/api': {
              target: apiProxyTarget,
              changeOrigin: true,
            },
          }
        : undefined,
    },
    build: {
      target: 'esnext',
    },
  };
});
