import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3000,  // Using port 3001 as it's currently running
    host: '0.0.0.0',
    hmr: {
      port: 3000,
    },
  },
  build: {
    target: 'esnext',
  },
});
