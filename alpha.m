fid = fopen('bigsynth_alpha.dat', 'a+');

for i=1:20000
  f = string(i)+"_cluster.txt";
  x = readmatrix(f);

  shp = alphaShape(x, 0.1);
  thr = criticalAlpha(shp, 'one-region');

  step = 0.1;
  nreg = 2;

  while nreg ~=1
    shp = alphaShape(x, thr);
    nreg = numRegions(shp);
    thr = thr + step ;
  end

  fprintf(fid, '%s \t %f \t %f \n', f, volume(shp), surfaceArea(shp));
end
fclose(fid);
