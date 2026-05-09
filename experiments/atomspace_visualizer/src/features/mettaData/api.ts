import { fetchText } from '../../shared/api/http';
import { env } from '../../shared/config/env';

export const loadMettaDataset = () => fetchText(`${env.apiBaseUrl}/api/data.metta?t=${Date.now()}`);
