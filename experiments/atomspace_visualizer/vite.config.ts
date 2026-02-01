import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'https://mindplex-hyperon-api-664ofptz2q-uc.a.run.app/',
        changeOrigin: true,
      },
    },
    hmr: {
      clientPort: 443,
    },
  },
  build: {
    target: 'esnext',
  },
});
