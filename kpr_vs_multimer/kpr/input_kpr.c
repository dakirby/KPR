/* define input variables as a function of step, which cell, and n-attempts = loop [0,ntries)
set history[inputSpecies][time][ncell] to correct value.
*/


void inputs(int time,int ncell,int n_attempts){

   int inputSpecies = trackin[0];

   history[inputSpecies][time][ncell] = isignal[time][ncell];
}
