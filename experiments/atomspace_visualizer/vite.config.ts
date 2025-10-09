import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3001,  // Using port 3001 as it's currently running
    host: '0.0.0.0',
    hmr: {
      port: 3001,
      host: 'urban-potato-v6gr5vqg6559fpqrg-3001.app.github.dev', // Use GitHub Codespaces URL only
    },
  },
  build: {
    target: 'esnext',
  },
});
