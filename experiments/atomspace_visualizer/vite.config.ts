import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'https://super-duper-winner-97q9rxp6vvx9hxq5q-5000.app.github.dev',
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
