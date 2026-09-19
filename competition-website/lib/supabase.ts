
import { createClient } from '@supabase/supabase-js';

// These should be environment variables.
// For now, we leave them as empty strings or placeholders.
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
