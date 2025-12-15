import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'https://mindplex-hyperon-3.onrender.com',
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
