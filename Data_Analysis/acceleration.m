function [acc_cam, acc_opti, time_cam, time_opti] = acceleration(db_cam, db_opti)

% rufe Funktion mit sc1_data_person1_preprocessed.csv und
% sc1_data_person1_optitrackprocessed.csv auf. Dann plotte das mit der
% jeweiligen Zeit. Komisches Ergebnis.
time_cam = db_cam(11:110,4)-27,74;
pitch_cam = db_cam(11:110,5);

time_opti = db_opti(444:5672,1);
pitch_opti = db_opti(444:5672,2);


pitch_cam_dt = gradient(pitch_cam, time_cam);
pitch_cam_dt2 = gradient(pitch_cam_dt, time_cam);

pitch_opti_dt = gradient(pitch_opti, time_opti);
pitch_opti_dt2 = gradient(pitch_opti_dt, time_opti);

acc_cam = pitch_cam_dt2;
acc_opti = pitch_opti_dt2;

end

