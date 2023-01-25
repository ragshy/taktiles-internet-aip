function [eul1, eul2] = quat_to_euler(db)
quat2 = db(:,3:6);      % speaker 2
quat1 = db(:,19:22);    % speaker 1


eul2 = quat2eul(quat2);
eul1 = quat2eul(quat1);
end
