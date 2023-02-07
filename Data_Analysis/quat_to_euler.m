function [eul1,eul2, db_pp1, db_pp2] = quat_to_euler(db)
% speaker 2
quat2 = db(:,3:6);   
% rearrange quaternion to wxyz
w2 = quat2(:, 4);
quat2 = quat2(:,[1:3]);
quat2 = [w2, quat2];
eul2 = quat2eul(quat2);
eul2 = rad2deg(eul2);

% Match / Sort Euler-ZYX to Pitch, Roll, Yaw
idx = [3 1 2];
eul2 = eul2(:,idx);

% speaker 1
quat1 = db(:,19:22);    
w1 = quat1(:, 4);
quat1 = quat1(:,[1:3]);
quat1 = [w1, quat1];
eul1 = quat2eul(quat1);
eul1 = rad2deg(eul1);
eul1 = eul1(:,idx);

% rearranging processed optitrack database to python database
% add eul1 and eul2
db_pp = db;
db_pp(:,3:5) = eul2;
db_pp(:,19:21) = eul1;
% drop unneeded columns 
db_pp = db_pp(:,[2:5, 7:9, 19:21, 23:25]);
db_pp = db_pp(5:end,:);
db_pp1 = db_pp(:,[1,8:end]);
db_pp2 = db_pp(:,1:7);

data_person1 = array2table(db_pp1);
data_person1.Properties.VariableNames(1:7) = {'Time','Pitch','Roll','Yaw','X-Pos','Y-Pos','Z-Pos'};
writetable(data_person1,'sc_data_person1_optitrackprocessed.csv');

data_person2 = array2table(db_pp2);
data_person2.Properties.VariableNames(1:7) = {'Time','Pitch','Roll','Yaw','X-Pos','Y-Pos','Z-Pos'};
writetable(data_person2,'sc_data_person2_optitrackprocessed.csv');
end
